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

# os.environ["LOCAL_RANK"]='0'
# os.environ["RANK"]='0'
# os.environ["WORLD_SIZE"]='4'
# os.environ["MASTER_ADDR"]='127.0.0.1'
# os.environ["MASTER_PORT"]='29500'

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='PyTorch GPT2 pipeline ft script')
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

    def set_pipeline(self, device_ids:list,config):
        
        # setup p2p subgroups
        # Pytorch does not support nccl send and recv, we mimic the behavior by broadcast within group of 2 
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
    def __init__(self, config):
        super(GPT2PipeLMModel, self).__init__()
        self.config=config
        self.transformer = GPT2Stage(config)
        
        # How to communicate this part? 
        # 1. board_cast in first stage, use the prev_group
        self.lm_head = GPT2PipeLMModel(self.transformer.wte.weight, config) # head has to be on the same device as the embedding
        self.apply(self._init_weights)
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

def main():
    
    args = parser.parse_args()

    # TODO: Fill for sing and k8s
    # torch.distributed init
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    
    world_size = torch.distributed.get_world_size()
    world = [i for i in range(world_size)]
    output_device = world_size-1

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
        sampler=torch.utils.data.SequentialSampler(train_data)
    )
    
    valid_loader = DataLoader(
        valid_data, batch_size=args.valid_batch_size, num_workers=0, 
        shuffle=False, pin_memory=False, drop_last=False,
        sampler=torch.utils.data.SequentialSampler(valid_data)
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
    model = GPT2Stage(pipes[local_rank],local_rank,world_size,config)    
    model.set_pipeline(world,config)
    for idx, data in enumerate(train_loader):
        data = {key: value for key, value in data.items()}
        _input = data['input'].to(local_rank)
        # _target = data['target'].to(args.device)
        # _msk = data['mask'].to(args.device)
        
        hidden_states, presents = model(_input)
        if local_rank == output_device:
            print(presents.shape)
        
    # Make different groups to mimic p2p
    # p2p_groups = {}
    # for r in range(world_size-1):
    #     p2p_groups[r] = (dist.new_group([r,r+1]))
    # p2p_groups[world_size-1]=(dist.new_group([world_size-1,0]))
        
    # test_tensor = torch.tensor([0,0,0,0],device=rank)
    # if rank==0: 
    #     test_tensor = torch.tensor([1,1,1,1],device=rank)
    #     dist.broadcast(test_tensor,0,group=p2p_groups[0]) # send
    #     dist.broadcast(test_tensor,3,group=p2p_groups[3]) # recv
    #     print(f'rank {rank}, value: {test_tensor}',flush=True)
    # elif rank==1:
    #     dist.broadcast(test_tensor,0,group=p2p_groups[0]) # recv
    #     print(f'rank {rank}, value: {test_tensor}',flush=True)
    #     test_tensor +=1
    #     dist.broadcast(test_tensor,1,group=p2p_groups[1]) # send
    # elif rank==2:
    #     dist.broadcast(test_tensor,1,group=p2p_groups[1]) # recv
    #     print(f'rank {rank}, value: {test_tensor}',flush=True)
    #     test_tensor +=1
    #     dist.broadcast(test_tensor,2,group=p2p_groups[2]) # send
    # elif rank==3:
    #     dist.broadcast(test_tensor,2,group=p2p_groups[2]) # recv
    #     print(f'rank {rank}, value: {test_tensor}',flush=True)
    #     test_tensor +=1
    #     dist.broadcast(test_tensor,3,group=p2p_groups[3]) # send

    # if rank==0: 
    #     test_tensor = torch.tensor([1,1,1,1],device=rank)
    #     dist.send(test_tensor,1,group=p2p_groups[0]) # send
    #     dist.recv(test_tensor,3,group=p2p_groups[3]) # recv
    #     print(f'rank {rank}, value: {test_tensor}',flush=True)
    # elif rank==1:
    #     dist.recv(test_tensor,0,group=p2p_groups[0]) # recv
    #     print(f'rank {rank}, value: {test_tensor}',flush=True)
    #     test_tensor +=1
    #     dist.send(test_tensor,2,group=p2p_groups[1]) # send
    # elif rank==2:
    #     dist.recv(test_tensor,1,group=p2p_groups[1]) # recv
    #     print(f'rank {rank}, value: {test_tensor}',flush=True)
    #     test_tensor +=1
    #     dist.send(test_tensor,3,group=p2p_groups[2]) # send
    # elif rank==3:
    #     dist.recv(test_tensor,2,group=p2p_groups[2]) # recv
    #     print(f'rank {rank}, value: {test_tensor}',flush=True)
    #     test_tensor +=1
    #     dist.send(test_tensor,0,group=p2p_groups[3]) # send
    
main()

