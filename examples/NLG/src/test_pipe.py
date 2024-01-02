import argparse
import time
import math
import os, sys
import itertools

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.distributed.rpc as rpc
from torch.nn.parameter import Parameter
import loralib as lora
import copy


from data_utils import FT_Dataset
from torch.utils.data import DataLoader
import json
from model import GPT2Config
from model import Block, LayerNorm
import random
import torch.distributed.rpc as rpc
rpc.init_rpc()

from gpu import (
    add_gpu_params, 
    parse_gpu, 
    distributed_opt, 
    distributed_gather, 
    distributed_sync, 
    cleanup
)

from optimizer import (
    create_adam_optimizer, 
    create_optimizer_scheduler, 
    add_optimizer_params, 
    create_adam_optimizer_from_args
)

# os.environ["LOCAL_RANK"]='0'
# os.environ["RANK"]='0'
# os.environ["WORLD_SIZE"]='4'
# os.environ["MASTER_ADDR"]='127.0.0.1'
# os.environ["MASTER_PORT"]='29500'

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='PyTorch GPT2 pipeline ft script')

add_gpu_params(parser)
add_optimizer_params(parser)

parser.add_argument('--train_data', required=True, help='location of training data corpus')

parser.add_argument('--valid_data', required=True, help='location of validation data corpus')

parser.add_argument('--train_batch_size', type=int, default=8, help='training batch size')

parser.add_argument('--valid_batch_size', type=int, default=4, help='validation batch size')

parser.add_argument('--grad_acc', type=int, default=1, help='gradient accumulation steps')

parser.add_argument('--clip', type=float, default=0.0, help='gradient clip')

parser.add_argument('--seq_len', type=int, default=512, help='number of tokens to predict.')

parser.add_argument('--model_card', default='gpt2.md', choices=['gpt2.sm', 'gpt2.md', 'gpt2.lg'], 
                    help='model names')

parser.add_argument('--init_checkpoint', default=None, help='pretrained checkpoint path')

parser.add_argument('--fp16', action='store_true', help='train model with fp16')

parser.add_argument('--log_interval', type=int, default=100, help='log interval')

parser.add_argument('--eval_interval', type=int, default=2000, help='eval interval')

parser.add_argument('--save_interval', type=int, default=500, help='save interval')

parser.add_argument('--work_dir', type=str, default=os.getenv('PT_OUTPUT_DIR', 'gpt2_model'), 
                    help='working folder.')

parser.add_argument('--lora_dim', type=int, default=0, help='lora attn dimension')

parser.add_argument('--lora_alpha', type=int, default=128, help='lora attn alpha')

parser.add_argument('--obj', default='clm', choices=['jlm', 'clm'], 
                    help='language model training objective')

parser.add_argument('--lora_dropout', default=0.0, type=float, 
                    help='dropout probability for lora layers')

parser.add_argument('--label_smooth', default=0.0, type=float, help='label smoothing')

parser.add_argument('--roll_interval', type=int, default=-1, help='rolling interval')

parser.add_argument('--roll_lr', type=float, default=0.00001, help='rolling learning rate')

parser.add_argument('--roll_step', type=int, default=100, help='rolling step')

parser.add_argument('--eval_epoch', type=int, default=1, help='eval per number of epochs')

parser.add_argument('--pipeline', type=int, default=1, help='pipeline parallel')

parser.add_argument('--gpu', type=json.loads, default=[0], help='pipeline parallel')

parser.add_argument('--partitions', type=json.loads, default=None, help='pipeline partition')

parser.add_argument('--ddp', type=bool, default=False, help='pipeline partition')

def print_args(args):
    if args.rank == 0:
        print('=' * 100)
        for k, v in args.__dict__.items():
            print(f'        - {k} : {v}')
        print('=' * 100)
        
class MultiplyStage(nn.Module):
    def __init__(self,device,config,multiplier):
        super(MultiplyStage,self).__init__()
        self.config = config
        self.device = device
        self.prev_device = device - 1 if device >0 else world_size-1
        self.next_device = device + 1 if device < world_size-1 else 0
        self.world_size = world_size    
        self.ld_map = []
        self.multiplier = multiplier
        self.ln = nn.Linear(1,1).to(device)
    def set_pipeline(self, device_ids:list,config, p2p_groups=None):
        
        # setup p2p subgroups
        # Pytorch does not support nccl send and recv, we mimic the behavior by broadcast within group of 2 
        if p2p_groups is not None: 
            self.p2p_groups = p2p_groups
        else:
            self.p2p_groups = {}
            for r in range(config.world_size-1):
                self.p2p_groups[r] = (torch.distributed.new_group([r,r+1]))
            self.p2p_groups[config.world_size-1]=(torch.distributed.new_group([config.world_size-1,0]))
        
    def forward(self, x):
        return self.ln(x)
class TestModel(nn.Module):
    def __init__(self,device_ids,device, config, world_size=1,pipeline=True):
        super(TestModel,self).__init__() 
        self.layer=MultiplyStage(device,config,device+1)
        if pipeline: 
            # cur_device = int(os.environ["LOCAL_RANK"])
            self.p2p_groups = {}
            for r in range(world_size-1):
                self.p2p_groups[r] = (dist.new_group([r,r+1]))
            self.p2p_groups[world_size-1] = (dist.new_group([world_size-1,0]))
        
    def forward(self,x):
        cur_device = int(os.environ["LOCAL_RANK"])
        intermediate  = torch.zeros(1,device=cur_device)
        if cur_device==0: 
            print(f'Input {x}')
            intermediate = self.layer(x)
        if cur_device != 0:
            dist.broadcast(intermediate,cur_device-1,group=self.p2p_groups[cur_device-1]) #recv
            print(f'Stage  {cur_device} recv {intermediate}, group={self.p2p_groups[cur_device]}')
            intermediate = self.layer(intermediate)
        if cur_device != 3: 
            dist.broadcast(intermediate,cur_device,group=self.p2p_groups[cur_device]) #send
            print(f'Stage  {cur_device} send {intermediate}, group={self.p2p_groups[cur_device]}')
            
        # if cur_device == 3: 
        #     return intermediate
        return intermediate

  
class AverageMeter(object):
    """Computes and stores the average and current value
         Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def optimizer_step(_loss, _optimizer, _model, _schedule, args, is_update=True):
    if args.fp16:
        with amp.scale_loss(_loss, _optimizer) as _scaled_loss:
            _scaled_loss.backward()
    else: 
        _loss.backward()
    
    if is_update: 
        if args.clip > 0:
            if args.fp16:
                torch.nn.tuils.clip_grad_norm_(amp.master_params(_optimizer), args.clip)
            else: 
                torch.nn.utils.clip_grad_norm_(_model.parameters(), args.clip)
        
        _optimizer.step()
        _optimizer.zero_grad()
    
    if _schedule is not None: 
        _schedule.step()
        

def evaluate(model, valid_loader, args):
    model.eval()
    total_loss = 0.
    start_time = time.time()
    
    avg_lm_loss = AverageMeter()
    
    with torch.no_grad():
        for idx, data in enumerate(valid_loader):
            data = {key: value for key, value in data.items()}
            
            _input = data['input'].to(args.device)
            _target = data['target'].to(args.device)
            _msk = data['mask'].to(args.device)
            
            _lm_logits, _loss = model(_input, lm_labels=_target, lm_mask=_msk)
            loss = _loss.mean()
            
            avg_lm_loss.update(loss.item())
            
            if idx % 100 == 0: 
                print('eval samples:', idx, 'loss', loss.float())
                
        total_time = time.time() - start_time 
        print('average loss', avg_lm_loss)
    return avg_lm_loss.avg, math.exp(avg_lm_loss.avg)

        
def train_validate(
    model,
    optimizer, 
    scheduler, 
    train_loader,
    valid_loader, 
    args,
    train_step=0,
    epoch=0
):
    loss_fct = nn.CrossEntropyLoss(ignore_index=1,reduce=False)
    
    model.train()
    avg_lm_loss = AverageMeter()
    print('start to train the model............', epoch)
    log_start_time = time.time()
    best_val_ppl = None
    
    for idx, data in enumerate(train_loader):
        data = {key: value for key, value in data.items()}
        
        _input = data['input'].to(args.device)
        _target = data['target'].to(args.device)
        _msk = data['mask'].to(args.device)

        _lm_logits, _lm_loss = model(
            _input, lm_labels=_target, lm_mask=_msk, label_smooth=args.label_smooth
        )
        
        
        _lm_loss = _lm_loss.mean()
        
        train_step += 1
        is_update = True if train_step % args.grad_acc == 0 else False 
        avg_lm_loss.update(_lm_loss.item())
        optimizer.step(
            _lm_loss/(args.grad_acc), optimizer, model, scheduler, args, is_update=is_update
        )
        
        if train_step % args.save_interval == 0:
            if args.rank == 0:
                model_path = os.path.join(args.work_dir, f'model.{train_step}.pt')
                print(f'saving checkpointing {model_path}')
                torch.save({'model_state_dict': {lora.lora_state_dict(model)}}, model_path)
            args.dist.barrier()
        
        # evaluation interval
        if train_step % args.eval_interval == 0: 
            eval_start_time = time.time()
            
            valid_loss, valid_ppl = evaluate(model, valid_loader, args)
            
            if best_val_ppl is None or valid_ppl < best_val_ppl:
                best_val_ppl = valid_ppl
                
            log_str = f'| Eval {train_step // args.eval_interval:3d} at step {train_step:>8d} | '\
                      f'time: {time.time() - eval_start_time:5.2f}s | valid loss {valid_loss:5.2f} | ' \
                      f'valid ppl {valid_ppl:5.2f} | best ppl {best_val_ppl:5.2f} '
            
            if args.rank == 0:
                print('-' * 100)
                print(log_str)
                print('-' * 100)
                
            model.train()
            args.dist.barrier()
        
        if train_step == args.max_step:
            break
        
    if args.rank == 0: 
        model.path = os.path.join(args.work_dir, f'model.{train_step}.pt')
        print('saving checkpoint', model_path)
        torch.save({'model_state_dict': model.state_dict()}, model_path)
    args.dist.barrier()
    return train_step
             
if __name__ == '__main__':
    args = parser.parse_args()

    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    
    args.dist = dist
    world_size = torch.distributed.get_world_size()
    world = [i for i in range(world_size)]
    output_device = world_size-1

    # print_args(args)

    train_data = FT_Dataset(
        args.train_data, args.train_batch_size, args.seq_len, 
        joint_lm=args.obj=='jlm'
    )     
    
    valid_data = FT_Dataset(
        args.valid_data, args.valid_batch_size, args.seq_len,
    )

    train_loader = DataLoader(
        train_data, batch_size=args.train_batch_size, num_workers=0, 
        shuffle=False, pin_memory=False, drop_last=True,
        sampler=torch.utils.data.RandomSampler(train_data)
    )
    
    valid_loader = DataLoader(
        valid_data, batch_size=args.valid_batch_size, num_workers=0, 
        shuffle=False, pin_memory=False, drop_last=False,
        sampler=torch.utils.data.RandomSampler(valid_data)
    )


    config = GPT2Config(
        n_embd=768, n_layer=12, n_head=12, 
        lora_attn_dim=args.lora_dim, 
        lora_attn_alpha=args.lora_alpha, 
        lora_dropout=args.lora_dropout,
        partitions=args.partitions,
        world_size=world_size
    )

    model = TestModel(world, local_rank, config, world_size,pipeline=True)
    # print(f'Current arch on GPU {local_rank}: {model}')
    
    # model.set_pipeline(world,config)
        
    args.lr = 1.0

    optimizer = create_adam_optimizer_from_args(model, args) # distribtued opt? 
    loss_fct = nn.L1Loss().to(local_rank)
    target = torch.tensor([0.5]).to(local_rank)

    input = torch.tensor([1.0]) 
    input = input.to(local_rank)
    model.train()
    output = model(input)
    # print(f'Before opt {local_rank}:') 

    for n,p in model.named_parameters():
        print(f'(stage {local_rank}){n,p.item()}')

    if local_rank == output_device:
        loss = loss_fct(output,target)
        print(f'rank {local_rank}, output={output}, loss={loss}')

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    # print(f'\nAfter opt Stage {local_rank}:') 
    # for n,p in model.named_parameters():
    #     print(f'(stage {local_rank}){n,p.item()}')
    #     print(n,p.item())

    
    dist.barrier()
    print('cleanup dist ...')
    cleanup(args)
    
    
    
    
    

