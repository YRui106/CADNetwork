# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 09:53:05 2024

@author: Administrator
"""

import numpy as np
import torch
from torch import flatten, nn
from torch.nn import init
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn import functional as F
# import math


class FTCA(nn.Module):

    def __init__(self, dim=512,kernel_size=3,attn_drop=0,agent_num=36):
        super().__init__()
        self.dim=dim
        self.kernel_size=kernel_size
        self.agent_num = agent_num

        self.pool_size = int(agent_num ** 0.5)

        self.key_embed=nn.Sequential(
            nn.Conv2d(dim,dim,kernel_size=kernel_size,padding=kernel_size//2,groups=3,bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )

        self.value_embed = nn.Conv2d(dim, dim, 1, 1, 0, bias = False)

        factor=4
        self.attention_embed=nn.Sequential(
            nn.Conv2d(2*agent_num,2*agent_num//factor,1,bias=False),
            nn.BatchNorm2d(2*agent_num//factor),
            nn.ReLU(),
            nn.Conv2d(2*agent_num//factor,kernel_size*kernel_size*agent_num,1)
        )
       
        self.softmax = nn.Softmax(dim=-1)
        
        # agent token
 
        self.pool = nn.AdaptiveAvgPool2d(self.pool_size)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        bs,c,h,w=x.shape
        k1=self.key_embed(x) #bs,c,h,w
        agent_tokens = self.pool(x)   #尝试取q1和k1的比例和
        # print('agent_tokens.shape',agent_tokens.shape)
        v=self.value_embed(x)#bs,c,h,w
        v = self.pool(v).view(bs,c,-1)
        k1 = k1.view(bs,-1,c)
        q1 = x.view(bs,-1,c)
        agent_tokens = agent_tokens.view(bs,-1,c)
        agent_attn = self.softmax(agent_tokens @ k1.transpose(-2, -1))
        agent_attn = self.attn_drop(agent_attn)
        q_attn = self.softmax(q1 @ agent_tokens.transpose(-2, -1))
        q_attn = self.attn_drop(q_attn).transpose(-2, -1)
        y=torch.cat([agent_attn,q_attn],dim=1) #bs,2c,h,w
        y = y.view(bs,-1,h,w)
        att=self.attention_embed(y) #bs,c*k*k,h,w
        att=att.reshape(bs,self.agent_num,self.kernel_size*self.kernel_size,h,w)
        att=att.mean(2,keepdim=False).view(bs,-1,self.agent_num) #bs,c,h*w
        k2=F.softmax(att,dim=-1)
        k2 =  k2 @ v.transpose(-2,-1)
        k2 =  k2.transpose(-2,-1)
        out = k2.view(bs,c,h,w)+k1.view(bs,c,h,w)

        return out


if __name__ == '__main__':
    input=torch.randn(50,512,7,7)
    cot = FTCA(dim=512,kernel_size=3)
    output=cot(input)
    print(output.shape)