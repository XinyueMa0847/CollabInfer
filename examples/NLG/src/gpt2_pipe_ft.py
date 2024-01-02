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
        
class GPT2Stage(nn.Module):
    def __init__(self, layer_nos,device,world_size,config):
        super(GPT2Stage, self).__init__()
        self.n_layer = config.n_layer
        self.n_embd = config.n_embd
        self.n_vocab = config.vocab_size

        self.layer_st, self.layer_last = layer_nos
        # TODO: load balancing by compute and memory demand 
        if self.layer_st == 0: # embedding layer to first stage
            self.wte = nn.Embedding(config.vocab_size, config.n_embd).to(device)
            self.wpe = nn.Embedding(config.n_positions, config.n_embd).to(device)
        elif self.layer_last == self.n_layer: 
            self.wte = nn.Embedding(config.vocab_size, config.n_embd).to(device)
            self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon).to(device)

        block = Block(config.n_ctx, config, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block).to(device) for _ in range(self.layer_st, self.layer_last)])

        self.config = config
        self.device = device
        self.prev_device = device - 1 if device >0 else world_size-1
        self.next_device = device + 1 if device < world_size-1 else 0
        self.world_size = world_size    
        self.ld_map = []

    def get_past(self, past=None, len_past=None):
        if past is None: 
            past_length = 0
            past = [None] * len(self.h)
        elif len_past is None:
            # equal size for past. []
            past_length = past[0][0].size(-2)
        return past, past_length
    def pos(self,position_ids, input_ids, len_past,past_length):
        # Generate from INPUT_IDS
        if position_ids is None and len_past is None: 
            position_ids = torch.arange(
                past_length, input_ids.size(-1) + past_length, 
                dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        #  Mathcing with LEN_PAST 
        elif len_past is not None:
            position_ids = (len_past).unsqueeze(1) #.long()
        # Else, do nothing
        return position_ids

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
        
        # Semantic and Sanity checks 
        if config.partitions is not None:
            self.partitions = config.partitions
            if not len(self.partitions ) == len(device_ids)+1: raise Exception(f"pipeline partitions '--partitions' {len(self.partitions )} more than num of devices {len(device_ids)}")
            if not self.partitions[-1] == self.n_layer : raise Exception("pipeline partitions '--partitions' > exceed number of layers in the model")
        else: self.partitions = [self.n_layer]
   
    def forward(
        self, 
        input_ids, 
        position_ids=None, 
        token_type_ids=None, 
        past=None, 
        len_past=None
    ):
    
        batch_size, seq_len = input_ids.shape
        hidden_shape = (batch_size,seq_len,self.n_embd)
        
        # Initialize PAST, used in every stage
        past, past_length = self.get_past(past,len_past)
        presents=None

        if self.device == 0: # first stage, forward first

            # Initialize position_ids 
            position_ids = self.pos(position_ids,input_ids,len_past,past_length)

            # Embeddings 
            inputs_embeds = self.wte(input_ids.view(-1, input_ids.size(-1)))     
            position_embeds = self.wpe(position_ids.view(-1, position_ids.size(-1)))

            if token_type_ids is not None:
                token_type_embeds = self.wte(token_type_ids.view(-1, token_type_ids.size(-1)))
            else:
                token_type_embeds = 0
                
            # Hiddent States
            hidden_states = inputs_embeds + position_embeds + token_type_embeds    
            for block, layer_past in zip(self.h, past):
                hidden_states, present = block(hidden_states, layer_past = layer_past, len_past=len_past)
                # present <2, batch, head, n_emd, head_features:64, figure later>
                if presents == None: 
                    presents = present.unsqueeze(0)
                else: 
                    presents = torch.cat((presents,present.unsqueeze(0)))
        
        if self.device > 0 : # recv, then forward 
            # intermediate tensors 
            hidden_states = torch.zeros(hidden_shape,device=self.device)
            presents  = torch.zeros((self.layer_st,2,batch_size,self.config.n_head,seq_len,64),device=self.device)
            
            # recv from prev stage
            dist.broadcast(hidden_states,self.prev_device,group=self.p2p_groups[self.prev_device])
            dist.broadcast(presents,self.prev_device,group=self.p2p_groups[self.prev_device])
            
            for block, layer_past in zip(self.h, past):
                hidden_states, present = block(hidden_states, layer_past = layer_past, len_past=len_past)
                presents = torch.cat((presents,present.unsqueeze(0)))
                    
        if self.device < self.world_size-1: # send to next stage  
            dist.broadcast(hidden_states,self.device,group=self.p2p_groups[self.device]) # send
            dist.broadcast(presents,self.device,group=self.p2p_groups[self.device]) # send

        if self.device == self.world_size - 1: # Last stage, post processing
            hidden_states = self.ln_f(hidden_states)
            output_shape = input_ids.size() + (hidden_states.size(-1),)
            return hidden_states.view(*output_shape), presents
        else: 
            return None, None

class GPT2LMHead(nn.Module):
    def __init__(self, model_embeddings_weights, config):
        super(GPT2LMHead, self).__init__()
        self.n_embd = config.n_embd
        self.set_embeddings_weights(model_embeddings_weights) # unnecessary at the init stage
        self.device = 0 

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embeddings_weights  # Tied weights

    def set_pipeline(self, device_ids):
        # Heads are always on the last partition
        # Put layers to different devices 
        self.device = device_ids[0]
        self.decoder.to(self.device)
        
    def forward(self, hidden_state):
        # Truncated Language modeling logits (we remove the last token)
        # h_trunc = h[:, :-1].contiguous().view(-1, self.n_embd)
        
        lm_logits = self.decoder(hidden_state)
        return lm_logits
class IdentityStage(nn.Module):
    def __init__(self,device):
        super(IdentityStage,self).__init__()
        self.ln1 = nn.Identity(10).to(device)
    def forward(self, x):
        return(self.ln1(x)+1)

class TestModel(nn.Module):
    def __init__(self,device_ids,world_size=1,pipeline=True):
        super(TestModel,self).__init__() 
        self.layers =nn.ModuleList([IdentityStage(device) for device in device_ids])
        if pipeline: 
            cur_device = int(os.environ["LOCAL_RANK"])
            self.p2p_groups = {}
            for r in range(world_size-1):
                self.p2p_groups[r] = (dist.new_group([r,r+1]))
            print(self.p2p_groups)
    def _to_next_stage(self,x,cur_device):
        dist.broadcast(x,cur_device,group=self.p2p_groups[cur_device])
    def _from_prev_stage(self,x,prev_device):
        dist.broadcast(x,prev_device,group=self.p2p_groups[prev_device])
        
    def forward(self,x):
        cur_device = int(os.environ["LOCAL_RANK"])
        intermediate  = torch.zeros(10,device=cur_device)
        if cur_device==0: print(f'Input {x}')
        if cur_device != 0:
            dist.broadcast(intermediate,cur_device-1,group=self.p2p_groups[cur_device-1]) #recv
            print(f'Stage  {cur_device} recv {intermediate}, group={self.p2p_groups[cur_device-1]}')
        
        if cur_device != 3: 
            intermediate = self.layers[cur_device](intermediate)
            print(f'Stage  {cur_device} send {intermediate}, group={self.p2p_groups[cur_device]}')
            dist.broadcast(intermediate,cur_device,group=self.p2p_groups[cur_device]) #recv
            
        if cur_device == 3: 
            return intermediate

class GPT2PipeLMModel(nn.Module):
    def __init__(self, layer_nos,device,world_size, config):
        super(GPT2PipeLMModel, self).__init__()
        self.output_device = world_size-1
        self.config=config
        self.transformer = GPT2Stage(layer_nos,device,world_size,config)
        self.device = device
        # How to communicate this part? 
        # 1. board_cast in first stage, use the prev_group
        if self.device == self.output_device: 
            self.lm_head = GPT2LMHead(self.transformer.wte.weight, config).to(device) # head has to be on the same device as the embedding

        self.apply(self._init_weights)
    
    def set_tied(self):
        """ Make sure we are sharing the embeddings"""
        self.lm_head.set_embeddings_weights(self.transformer.wte.weight)
        
    def parse_pretrained_weight(self, state_dict):
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
    
        state_dict_tmp = copy.deepcopy(state_dict)
        
        old_keys = []
        new_keys = []
        for key in state_dict_tmp:
            new_key = None
            if key.endswith(".g"):
                new_key = key[:-2] + ".weight"
            elif key.endswith(".b"):
                new_key = key[:-2] + ".bias"
            elif key.endswith(".w"):
                new_key = key[:-2] + ".weight"
            
            if key.startswith("module.transformer."):
                new_key = key[len("module.transformer."):]

            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)

        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        # remove params for other stages 
        target_layers=[]
        layer_st, layer_last = self.transformer.layer_st, self.transformer.layer_last
        if self.device == 0:
            target_layers.extend(['wte','wpe'])
        if self.device == self.output_device:
            target_layers.extend(['wte'])
            target_layers.extend(['ln_f'])
        target_layers.extend([f'h.{x}' for x in range(layer_st,layer_last)] )
        
        for key in list(state_dict.keys()):
            if not key in target_layers:
                state_dict.pop(key)
    
        # keep the ones pretrained model does not contain
        for n, p in self.transformer.named_parameters():
            if n not in state_dict:
                state_dict[n] = p
        return state_dict
    
    def load_weight(self, state_dict):
        state_dict = self.parse_pretrained_weight(state_dict)
        print(f'Node {self.device}, keys={list((state_dict.keys()))}')
        self.transformer.load_state_dict(state_dict, strict=False)
        if self.device == self.output_device:
            self.set_tied()

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

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
        
        self.transformer.set_pipeline(device_ids,config, self.p2p_groups)
   

    def forward(
        self, 
        input_ids, 
        lm_labels=None, 
        lm_mask=None, 
        past=None, 
        len_past=None, 
        label_smooth=0.0,
        is_report_accuracy=False
    ):
        _batch, _len = input_ids.shape
        hidden_states, presents = self.transformer(input_ids, past=past, len_past=len_past) # pipelined, final results in last stage

        if self.device == self.output_device:
            lm_logits = self.lm_head(hidden_states)
            if lm_labels is not None: 
                
                if is_report_accuracy: 
                    _pred_token = torch.argmax(lm_logits, dim=-1)
                    _hit = (_pred_token == lm_labels) * lm_mask
                    
                    _t1_acc = torch.zeros(_batch, dtype=torch.float, device=self.device)
                    _all_acc = torch.zeros(_batch, dtype=torch.float, device=self.device)
                    
                    for _b in range(0, _batch):
                        for _i in range(0, _len):
                            if lm_mask[_b, _i] >= 1.0:
                                if _hit[_b, _i] > 0:
                                    _t1_acc[_b] = 1.0 
                                break 
                    
                    _is_succ = True 
                    for _i in range(0, _len):
                        if lm_mask[_b, _i] >= 1.0: 
                            if _hit[_b, _i] <= 0:
                                _is_succ = False 
                                break
                    
                    if _is_succ:
                        _all_acc[_b] = 1.0 
                    
                if label_smooth > 0.0001:
                    logprobs = torch.nn.functional.log_softmax(lm_logits.view(-1, lm_logits.size(-1)), dim=-1)
                    nll_loss = -logprobs.gather(dim=-1, index=lm_labels.view(-1).unsqueeze(1))
                    nll_loss = nll_loss.squeeze(1)
                    smooth_loss = -logprobs.mean(dim=-1)
                    loss = (1.0 - label_smooth) * nll_loss + label_smooth + smooth_loss
                    loss = loss.view(_batch, _len)
                else:
                    loss_fct = nn.CrossEntropyLoss(ignore_index=-1, reduce=False) 
                    loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1)).view(_batch, _len)
                
                if lm_mask is None: 
                    lm_mask = torch.ones(loss.shape, dtype=loss.dtype, device = self.device)
                loss = loss * lm_mask 
                
                loss = loss.sum() / (lm_mask.sum() + 0.0001)
                
                if is_report_accuracy:
                    return lm_logits, loss, _t1_acc, _all_acc
                else: 
                    return lm_logits, presents
        return None, presents
    
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
        
        # if 
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

    if args.fp16:
        try:
            from apex import amp
        except Exception as e:
            warnings.warn('Could not import amp, apex may not be installed')
            
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    
    # TODO: Fill for sing and k8s
    # torch.distributed init
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    
    world_size = torch.distributed.get_world_size()
    world = [i for i in range(world_size)]
    print(world)
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

    pipes = [[args.partitions[i],args.partitions[i+1]] for i in range(world_size)]
    model = GPT2PipeLMModel(pipes[local_rank], local_rank, world_size,config)
    print(f'Current arch on GPU {local_rank}: {model}')
    
    model.set_pipeline(world,config)
        
    if args.init_checkpoint is not None: 
        print('loading model pretrained weight.')
        model.load_weight(torch.load(args.init_checkpoint)) # Need to redesign
    
    if args.lora_dim > 0: 
        lora.mark_only_lora_as_trainable(model)
    optimizer = create_adam_optimizer_from_args(model, args) # distribtued opt? 
    
    if args.max_step is None: 
        print(args.max_epoch)
        print(train_data.num_batches)
        print(world_size)
        args.max_step = (args.max_epoch * train_data.num_batches + world_size -1) // world_size
        print('set max_step:', args.max_step)
        
    scheduler = create_optimizer_scheduler(optimizer, args)
    
    if args.fp16: 
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        
    try: 
        train_step = 0
        for epoch in itertools.count(start=1):
            train_step = train_validate(
                model, optimizer, scheduler, train_loader, valid_loader, args, 
                train_step=train_step, epoch=epoch
            )
            
            if train_step >= args.max_step or (args.max_epoch is not None and epoch >= args.max_epoch):
                if args.rank == 0:
                    print('-' * 100)
                    print('End of training')
                break 
    except KeyboardInterrupt:
        if args.rank == 0:
            print('-' * 100)
            print('Exiting from training early')
    
    dist.barrier()
    print('cleanup dist ...')
    cleanup(args)
    
    
    
    
    

