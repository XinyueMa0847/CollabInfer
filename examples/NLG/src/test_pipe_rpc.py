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
from torch.distributed.optim import DistributedOptimizer
import torch.distributed.autograd as dist_autograd
import threading
from gpu import (
    add_gpu_params, 
    parse_gpu, 
    distributed_opt, 
    distributed_gather, 
    distributed_sync, 
    cleanup
)

from optimizer import (
    AdamW,
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

class PipeStageBase(nn.Module):
    def __init__(self):
        super(PipeStageBase, self).__init__()
        self._lock = threading.Lock()
    def mark_only_lora_as_trainable(self,bias='none'):
        lora.mark_only_lora_as_trainable(self, bias=bias)

class GPT2HeadStage(PipeStageBase):
    def __init__(self,device,config):
        super(GPT2HeadStage, self).__init__()
        self.n_embd = config.n_embd
        
        self.device = device         
        self.config = config
        self.apply(self._init_weights) # decoder exampt
    def  _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
            
    def param_rrefs(self):
        # preserve name too? 
        return [rpc.RRef(p) for p in self.parameters()]
    
    def fetch(self, x):
        if x == None: return x 
        else: return x.to_here().to(self.device)
        
    def forward(self, hidden_states):
        hidden_states = self.fetch(hidden_states)
        with self._lock:
            lm_logits = self.decoder(hidden_states)
        return lm_logits        
    
    
    def set_embeddings_weights(self, model_embeddings_weights):
        model_embeddings_weights = self.fetch(model_embeddings_weights)
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embeddings_weights  # Tied weights
        for n,p in self.named_parameters():
            print(n)
            print(p.shape)
        
class GPT2EmbdStage(PipeStageBase):
    def __init__(self,device,config):
        super(GPT2EmbdStage, self).__init__()
        self.n_layer = config.n_layer
        self.n_embd = config.n_embd
        self.n_vocab = config.vocab_size
        
        self.device = device         
        self.config = config
        self.world_size = world_size    

        self.wte = nn.Embedding(config.vocab_size, config.n_embd).to(device)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd).to(device)
    
        self.apply(self._init_weights)
        # print(self)
        for n,p in self.named_parameters():
            print(n)
            print(p.shape)
    def  _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
            
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

    def param_rrefs(self):
        # preserve name too? 
        return [rpc.RRef(p) for p in self.parameters()]
    
    def fetch(self, x):
        if x == None: return x 
        else: return x.to_here().to(self.device)
    
    def send_wte_weight(self):
        w_rref = rpc.RRef(self.wte.weight)
        return w_rref
    def forward(
        self, 
        input_ids, 
        position_ids=None, 
        token_type_ids=None, 
        past=None, 
        len_past=None
    ):
        input_ids = self.fetch(input_ids)
        position_ids = self.fetch(position_ids)
        token_type_ids = self.fetch(token_type_ids)
        past=self.fecth(past)
        len_past = self.fetch(len_past)
        
        
        batch_size, seq_len = input_ids.shape
        hidden_shape = (batch_size,seq_len,self.n_embd)
        
        # Initialize PAST, used in every stage
        past, past_length = self.get_past(past,len_past)
        presents=None
    
        # Initialize position_ids 
        position_ids = self.pos(position_ids,input_ids,len_past,past_length)

        with self._lock:
            
            # Embeddings 
            inputs_embeds = self.wte(input_ids.view(-1, input_ids.size(-1)))     
            position_embeds = self.wpe(position_ids.view(-1, position_ids.size(-1)))

            if token_type_ids is not None:
                token_type_embeds = self.wte(token_type_ids.view(-1, token_type_ids.size(-1)))
            else:
                token_type_embeds = 0
                
            # Hiddent States
            hidden_states = inputs_embeds + position_embeds + token_type_embeds    

        return hidden_states 

class GPT2AttnStage(PipeStageBase):
    def __init__(self, layer_nos,device,world_size,config):
        super(GPT2AttnStage, self).__init__()
        self.n_layer = config.n_layer
        self.n_embd = config.n_embd
        self.n_vocab = config.vocab_size
        
        self.device = device         
        self.config = config
        self.world_size = world_size    
        self.layer_st, self.layer_last = layer_nos

        # elif self.device == world_size - 1:
        #      self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon).to(device)
        
        block = Block(config.n_ctx, config, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block).to(device) for _ in range(self.layer_st, self.layer_last)])

        self.apply(self._init_weights)
        # print(self)
        for n,p in self.named_parameters():
            print(n)
            print(p.shape)
    def  _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def param_rrefs(self):
        # preserve name too? 
        return [rpc.RRef(p) for p in self.parameters()]
    
    def fetch(self, x):
        if x == None: return x 
        else: return x.to_here().to(self.device)
        
    def forward(
        self, 
        hidden_states,
        presents,
        past, 
        len_past
    ):
        hidden_states = self.fetch(hidden_states)
        presents = self.fetch(presents)
        past=self.fecth(past)
        len_past = self.fetch(len_past)
        
        with self._lock:
        
            for block, layer_past in zip(self.h, past):
                hidden_states, present,_,_ = block(hidden_states, layer_past = layer_past, len_past=len_past)
                # present <2, batch, head, n_emd, head_features:64, figure later>
                if presents == None: 
                    presents = present.unsqueeze(0)
                else: 
                    presents = torch.cat((presents,present.unsqueeze(0)))
            
        return hidden_states, presents, past, len_past
    
class MultiplyStage(PipeStageBase):
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
    def param_rrefs(self):
        return [rpc.RRef(p) for p in self.parameters()]
    
    def forward(self, x_rref:rpc.RRef):
        x = x_rref.to_here().to(self.device)
        with self._lock:
            out=self.ln(x)
        return out

class TestModel(nn.Module):
    def __init__(self,device_ids,device, config, world_size=1,pipeline=True):
        super(TestModel,self).__init__() 
        self.last_device=device_ids[-1]
        self.layers=[rpc.remote(
            device_ids[i],
            MultiplyStage,
            args = (i,config,i+1)
        ) for i in device_ids]
        
    def forward(self,x):
        x_rref = rpc.RRef(x)
        for i in range(len(self.layers)-1):
            x_rref = self.layers[i].remote().forward(x_rref)
        y_rref = self.layers[-1].remote().forward(x_rref)
        # y_rref.to_here().to(self.last_device)
        print(y_rref)
        # return y_rref.to_here().to(self.device)
        return y_rref
        
    def param_rrefs(self):
        remote_params = []
        for layer in self.layers:
            remote_params.extend(layer.remote().param_rrefs().to_here())
        return remote_params

class GPT2Model(nn.Module):
    def __init__(self, devices, config):
        super(GPT2Model, self).__init__()
        self.config=config
        self.devices = devices
        self.world_size = len(devices)
        self.last_device = devices[-1]
        self.partitions = config.partitions
        
        self.embd_stage = rpc.remote(self.devices[0],GPT2EmbdStage,args = (self.devices[0],config))

        # attention layers 
        assert(len(self.partitions) == world_size+1)
        self.attns = []
        for i in range(world_size):
            self.attns.append(
                rpc.remote(devices[i],GPT2AttnStage, args = (self.partitions[i:i+2],devices[i],world_size,config))
            )
        self.ln_f = rpc.remote(self.last_device, LayerNorm, args = (config.n_embd, config.layer_norm_epsilon))
        self.heads = rpc.remote(self.last_device, GPT2HeadStage, args = (self.last_device, config))
        model_embeddings_weights = self.embd_stage.rpc_sync().send_wte_weight()
        self.heads.remote().set_embeddings_weights(model_embeddings_weights)
    def mark_only_lora_as_trainable(self,bias='none'):
        self.embd_stage.remote().mark_only_lora_as_trainable(bias=bias)
        for layer in self.attns:
            layer.remote().mark_only_lora_as_trainable(bias=bias)
        self.heads.remote().mark_only_lora_as_trainable(bias=bias)
        self.ln_f.remote().disable_grad()
    def set_tied(self): # assume calling from last device 
        model_embeddings_weights = self.embd_stage.rpc_sync().send_wte_weight()
        self.heads.remote().set_embeddings_weights(model_embeddings_weights)

    def param_rrefs(self):
        remote_params = []
        remote_params.extend(self.embd_stage.remote().param_rrefs().to_here())
        for layer in self.attns:
            remote_params.extend(layer.remote().param_rrefs().to_here())
        remote_params.extend(self.ln_f.remote().param_rrefs().to_here())
        remote_params.extend(self.heads.remote().param_rrefs().to_here())
            
        return remote_params
    
    def forward(
        self, 
        input_ids, 
        # lm_labels=None, 
        # lm_mask=None, 
        past=None, 
        len_past=None, 
        # label_smooth=0.0,
        # is_report_accuracy=False
    ):
    
        _batch, _len = input_ids.shape
        # make refs for inputs 
        input_rref = rpc.RRef(input_ids)
        past_rref = rpc.RRef(past)
        len_past_rref = rpc.RRef(len_past)
        
        # embd stage 
        hidden_states_rref = self.embd_stage.remote().forward(input_rref, past=past_rref, len_past=len_past_rref)
        
        # attns 
        presents_rref = rpc.RRef(None)
        for i in range(self.world_size):
            hidden_states_rref, presents_rref, past_rref, len_past_rerf = self.attns[i].remote().forward(
                hidden_states_rref, 
                presents=presents_rref, 
                past=past_rref, 
                len_past=len_past_rerf)
       
        hidden_states_rref = self.ln_f.remote().forward(hidden_states_rref)
       
        # heads
        lm_logits_rref = self.heads.remote().forward(hidden_states_rref)
        # if self.training: 
        #     return lm_logits_rref, presents_rref 
        # else:
        #     return lm_logits_rref, None
        return lm_logits_rref, None
        
        
    
        
        

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
            
            _batch, _len = _input.shape
            _lm_logits, _ = model(_input)
        
            _lm_logits = _lm_logits.local_value()
            loss = loss_fct(_lm_logits.view(-1, _lm_logits.size(-1)), _lm_logits.view(-1)).view(_batch, _len)

            if _msk is None: 
                _msk = torch.ones(loss.shape, dtype=loss.dtype, device=loss.device)
            loss = loss * _msk 
            loss = loss.sum() / (_msk.sum() + 0.001)
            
            avg_lm_loss.update(loss.item())
            
            if idx % 100 == 0: 
                print('eval samples:', idx, 'loss', loss.float())
                
        total_time = time.time() - start_time 
        print('average loss', avg_lm_loss)
    return avg_lm_loss.avg, math.exp(avg_lm_loss.avg)

class SmoothCrossEntropyLoss(nn.Module):
    def __init__(self, label_smooth, weight = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean') -> None:
        super(nn.Module, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.nll_loss = nn.NLLLoss(ignore_index=self.ignore_index,reduce=reduce,reduction=reduction)
        self.logprobs = nn.functional.log_softmax()
        self.label_smooth = label_smooth
    def forward(self, input, target):
        logprobs = self.logprobs(input.view(-1, target.size(-1)), dim=-1)       
        nll_loss = -logprobs.gather(dim=-1, index=target.view(-1).unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = (1.0 - self.label_smooth) * nll_loss + self.label_smooth * smooth_loss
        return loss
def loss_with_label_smooth(logits, labels, label_smooth,batch,len):
    logprobs = torch.nn.functional.log_softmax(logits.view(-1, logits.size(-1)), dim=-1)
    nll_loss=-logprobs.gather(dim=-1, index=labels.view(-1).unsqueeze(1))
    nll_loss = nll_loss.squeeze(1)
    smooth_loss = -logprobs.mean(dim=-1)
    loss =  (1.0 - label_smooth) * nll_loss + label_smooth * smooth_loss
    loss = loss.view(batch, len)
    return loss

def train_validate(
    model,
    optimizer, 
    scheduler, 
    train_loader,
    valid_loader, 
    args,
    train_step=0,
    epoch=0,
):
    if args.label_smooth:
        loss_fct = SmoothCrossEntropyLoss(args.label_smooth,ignore_index=-1, reduce=False)
    else:
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

        _batch, _len = _input.shape
        with dist_autograd.context() as context_id:

            _lm_logits_rref, _ = model(_input)
            _lm_logits = _lm_logits_rref.local_value()
            
            # Move loss calculation out 
            # _lm_loss = loss_fct(_lm_loss.view(-1, _lm_loss.size(-1)), _target.view(-1)).view(_batch, _len)
            
                    
            train_step += 1
            is_update = True if train_step % args.grad_acc == 0 else False 
            # avg_lm_loss.update(_lm_loss.item())
            # optimizer.step(
            #     _lm_loss/(args.grad_acc), optimizer, model, scheduler, args, is_update=is_update
            # )
        
            output = model(input)
            # print(f'Before opt {local_rank}:') 
            output = output.local_value().cpu()

            dist_autograd.backward(context_id, [loss_fct(_lm_logits.view(-1, _lm_logits.size(-1)), _target.view(-1)).view(_batch, _len)])
            # print(f'rank {local_rank}, output={output}')
            # grads = dist_autograd.get_gradients(context_id) # retrieve gradients

            # TODO if args.clip > 0, add clip fucntions here for RRef types 
            
            optimizer.step(context_id)
            if scheduler:
                scheduler.step()
                
            # opt.zero_grad() # distributed optimizer has no zero_grad
            # print(f'\nAfter opt Stage {local_rank}:') 
            # grads = dist_autograd.get_gradients(context_id) 
            # for param in model.param_rrefs():
            #     # print(f'{grad.item()}: {grads[grad]}')
            #     param=param.to_here()
            #     print(param)

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
            # args.dist.barrier()
        
        if train_step == args.max_step:
            break
        
    if args.rank == 0: 
        model.path = os.path.join(args.work_dir, f'model.{train_step}.pt')
        print('saving checkpoint', model_path)
        torch.save({'model_state_dict': model.state_dict()}, model_path)
    # args.dist.barrier()
    return train_step


if __name__ == '__main__':
    args = parser.parse_args()
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    
        
    
    
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    
    world_size = args.world_size
    world = [i for i in range(world_size)]
    output_device = world_size-1
    options = rpc.TensorPipeRpcBackendOptions()
    next = (local_rank+1)%world_size
    prev = (local_rank-1) if local_rank >0 else world_size-1
    
    # To support p2p directly
    # https://github.com/pytorch/pytorch/issues/53501 
    # https://github.com/pytorch/pytorch/blob/main/torch/distributed/rpc/options.py#L104
    if local_rank != world_size-1:
        options.set_device_map(f'rank{next}',{local_rank:next})
        options.set_device_map(f'rank{world_size-1}',{local_rank:world_size-1})
    else:
        options.set_device_map(f'rank{0}',{local_rank:0})
        options.set_device_map(f'rank{1}',{local_rank:1})
        # options.set_device_map(f'rank{2}',{local_rank:2})
        
    options.set_device_map(f'rank{prev}',{local_rank:prev})
    
    if local_rank != world_size-1:
        print(f'RPC_init rank{local_rank}')
        rpc.init_rpc(
            f'rank{local_rank}',
            rank=local_rank,
            world_size=world_size,
            rpc_backend_options=options
            )
        pass
    else:    
    
        print(f'RPC_init rank{local_rank}')
        rpc.init_rpc(
            f'rank{local_rank}',
            rank=local_rank,
            world_size=world_size,
            rpc_backend_options=options)
        
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


        if args.model_card == 'gpt2.sm':
            config = GPT2Config(
                n_embd=768, n_layer=12, n_head=12, 
                lora_attn_dim=args.lora_dim, 
                lora_attn_alpha=args.lora_alpha, 
                lora_dropout=args.lora_dropout,
                partitions=args.partitions
            )
        elif args.model_card == 'gpt2.md':
            config = GPT2Config(
                n_embd=1024, n_layer=24, n_head=16, 
                lora_attn_dim=args.lora_dim, 
                lora_attn_alpha=args.lora_alpha, 
                lora_dropout=args.lora_dropout,
                partitions=args.partitions
            )
        elif args.model_card == 'gpt2.lg':
            config = GPT2Config(
                n_embd=1280, n_layer=36, n_head=20, 
                lora_attn_dim=args.lora_dim, 
                lora_attn_alpha=args.lora_alpha, 
                lora_dropout=args.lora_dropout,
                partitions=args.partitions
            )


        # model = GPT2Model(world,config)

        # # if args.lora_dim > 0:
        # #     model.mark_only_lora_as_trainable()
        # opt = DistributedOptimizer(
        #     AdamW,
        #     params_rref=model.param_rrefs(),
        #     # params=[{"params": [p for n, p in model.named_parameters()]}] replaced
        #     lr=args.lr, 
        #     betas=(args.adam_beta1, args.adam_beta2), 
        #     eps=args.adam_epislon, 
        #     weight_decay=args.weight_decay, 
        #     correct_bias=args.correct_bias
        # )
        # for remote_opt in opt.remote_optimizers:
        #     remote = remote_opt.to_here()
        #     param_groups = remote.optim.param_groups[0]['params']
        #     print(len(param_groups))
        #     for param in param_groups:
        #         print(param.shape)
        from model import GPT2LMModel
        model2 = GPT2LMModel(config)
        print(model2)
        opt2 = create_adam_optimizer_from_args(model2,args)
        params = opt2.param_groups[0]['params']
        # print(len(params))
        for param in params:
            print(param.shape)
        for n,p in model2.named_parameters():
            print(n)
            print(p.shape)
        # for n, p in model2.lm_head.named_parameters():
        #     print(n)
        #     print(p.shape)
        
        if args.max_step is None:
            args.max_step = (args.max_epoch * train_data.num_batches + args.world_size - 1) // args.world_size
            print('set max_step:', args.max_step)
        scheduler = create_optimizer_scheduler(opt,args)
        print(scheduler)
        
        try:
            train_step = 0
            for epoch in itertools.count(start=1):
                train_step = train_validate(
                    model, opt, scheduler, train_loader, valid_loader, args, 
                    train_step=train_step, epoch=epoch
                )
                
                if train_step >= args.max_step or (args.max_epoch is not None and epoch >= args.max_epoch):
                    print('-' * 100)
                    print('End of training')
        except KeyboardInterrupt:
            print('-' * 100)
            print('Exiting from training early')
    print('rpc shutdown ...')
    rpc.shutdown()
    # cleanup(args)
    
    
    
    
    

