import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from fightingcv_attention.attention.CoTAttention import CoTAttention
from FTCA import FTCA

from einops import rearrange
from basicsr.archs.arch_util import flow_warp 
from basicsr.utils.registry import ARCH_REGISTRY
from thop import profile
import thop

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class ElementScale(nn.Module):
    """A learnable element-wise scaler."""

    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)),
            requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale

def ones(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0.0)

def batch_index_select(x, idx):
    if len(x.size()) == 3:
        B, N, C = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N, C)[idx.reshape(-1)].reshape(B, N_new, C)
        return out
    elif len(x.size()) == 2:
        B, N = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N)[idx.reshape(-1)].reshape(B, N_new)
        return out
    else:
        raise NotImplementedError

def batch_index_fill(x, x1, x2, idx1, idx2):
    B, N, C = x.size()
    B, N1, C = x1.size()
    B, N2, C = x2.size()

    offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1)
    idx1 = idx1 + offset * N
    idx2 = idx2 + offset * N

    x = x.reshape(B*N, C)

    x[idx1.reshape(-1)] = x1.reshape(B*N1, C)
    x[idx2.reshape(-1)] = x2.reshape(B*N2, C)

    x = x.reshape(B, N, C)
    return x

class Predictor(nn.Module):
    """ Importance Score Predictor
    """
    def __init__(self, dim, window_size=8, k=4,ratio=0.5):
        super().__init__()

        self.ratio = ratio
        self.window_size = window_size
        cdim = dim + k
        embed_dim = window_size**2
        
        self.in_conv = nn.Sequential(
            nn.Conv2d(cdim, cdim//4, 1),
            LayerNorm(cdim//4),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.out_offsets = nn.Sequential(
            nn.Conv2d(cdim//4, cdim//8, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(cdim//8, 2, 1),
        )

        self.out_mask = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim//16, 1),
            nn.Conv2d(embed_dim//16, 2, 1),
            nn.Softmax(dim=-1)
        )

        self.out_CA = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(cdim//4, dim, 1),
            nn.Sigmoid(),
        )

        self.out_SA = nn.Sequential(
            nn.Conv2d(cdim//4, 1, 3, 1, 1),
            nn.Sigmoid(),
        )        

        self.conv = nn.Conv2d(cdim//4,cdim//2,1,1)
    def forward(self, input_x, mask=None, ratio=0.5, train_mode=False):

        x = self.in_conv(input_x)

        offsets = self.out_offsets(x)
        offsets = offsets.tanh().mul(8.0)
        
        x3 = self.conv(x)
        x1,x2 = x3.chunk(2,dim=1)
        

        ca = self.out_CA(x1)
        sa = self.out_SA(x2)

        x = torch.mean(x, keepdim=True, dim=1) 

        x = rearrange(x,'b c (h dh) (w dw) -> b (dh dw c) h w', dh=self.window_size, dw=self.window_size)

        pred_score = self.out_mask(x)
        B, C, H, W = pred_score.size()
        pred_score = pred_score.view(B,H*W,C)
        
        mask = F.gumbel_softmax(pred_score, hard=True, dim=2)[:, :, 0:1]
        
        if self.training or train_mode:
            return mask, offsets, ca, sa
        else:
            score = pred_score[:, : , 0]
            B, N = score.shape
            r = torch.mean(mask,dim=(0,1))*1.0
            if self.ratio == 1:
                num_keep_node = N #int(N * r) #int(N * r)
            else:
                num_keep_node = min(int(N * r * 2 * self.ratio), N)
            idx = torch.argsort(score, dim=1, descending=True)  #argsort()返回数组元素排序后的索引值
            idx1 = idx[:, :num_keep_node]
            idx2 = idx[:, num_keep_node:]
            return [idx1, idx2], offsets, ca, sa


class CAMM(nn.Module):
    def __init__(self, dim, window_size=8, bias=True, is_deformable=True, ratio=0.5,agent_num=49,attn_drop=0):
        super().__init__()    

        self.dim = dim
        self.window_size = window_size
        self.is_deformable = is_deformable
        self.ratio = ratio
        self.kernel_size = 3

        self.agent_num = agent_num
        self.attn_drop = nn.Dropout(attn_drop)
        k = 3
        d = 2

        self.project_v = nn.Conv2d(dim, dim, 1, 1, 0, bias = bias)
        self.project_q = nn.Linear(dim, dim, bias = bias)
        self.project_k = nn.Linear(dim, dim, bias = bias)
        self.softmax = nn.Softmax(dim=-1)

        # Conv
        self.conv_sptial = nn.Sequential(
            nn.Conv2d(dim, dim, k, padding=k//2, groups=dim),
            nn.Conv2d(dim, dim, k, stride=1, padding=((k//2)*d), groups=dim, dilation=d))        
        self.project_out = nn.Conv2d(dim, dim, 1, 1, 0, bias = bias)

        self.act = nn.GELU()
        # Predictor
        self.route = Predictor(dim,window_size,ratio=ratio)

        self.pool_size = int(agent_num ** 0.5)
        self.pool = nn.AdaptiveAvgPool1d(self.pool_size)
        factor = 2
        self.attention_embed=nn.Sequential(
            nn.Conv2d(2*self.pool_size,2*self.pool_size//factor,1,bias=False),
            nn.BatchNorm2d(2*self.pool_size//factor),
            nn.ReLU(),
            nn.Conv2d(2*self.pool_size//factor,self.kernel_size*self.kernel_size*self.pool_size,1)
        )
    def forward(self,x,condition_global=None, mask=None, train_mode=False):
        N,C,H,W = x.shape
        h=H//self.window_size
        w=W//self.window_size
        v = self.project_v(x)

        if self.is_deformable:
            condition_wind = torch.stack(torch.meshgrid(torch.linspace(-1,1,self.window_size),torch.linspace(-1,1,self.window_size)))\
                    .type_as(x).unsqueeze(0).repeat(N, 1, H//self.window_size, W//self.window_size)
            if condition_global is None:
                _condition = torch.cat([v, condition_wind], dim=1)
            else:
                _condition = torch.cat([v, condition_global, condition_wind], dim=1)

        mask, offsets, ca, sa = self.route(_condition,ratio=self.ratio,train_mode=train_mode)
        #DFTCA
        q = x 
        k = x + flow_warp(x, offsets.permute(0,2,3,1), interp_mode='bilinear', padding_mode='border')
        qk = torch.cat([q,k],dim=1)
 
        sca =sa*ca
        vs=v*sca

        v  = rearrange(v,'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=self.window_size, dw=self.window_size)
        vs = rearrange(vs,'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=self.window_size, dw=self.window_size)
        qk = rearrange(qk,'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=self.window_size, dw=self.window_size)

        if self.training or train_mode:
            N_ = v.shape[1]
            v1,v2 = v*mask, vs*(1-mask)   
            qk1 = qk*mask 
        else:
            idx1, idx2 = mask
            _, N_ = idx1.shape
            v1,v2 = batch_index_select(v,idx1),batch_index_select(vs,idx2)
            qk1 = batch_index_select(qk,idx1)

        v1 = rearrange(v1,'b n (dh dw c) -> (b n) (dh dw) c', n=N_, dh=self.window_size, dw=self.window_size)
        
        qk1 = rearrange(qk1,'b n (dh dw c) -> b (n dh dw) c', n=N_, dh=self.window_size, dw=self.window_size)

        q1,k1 = torch.chunk(qk1,2,dim=2)
        q1 = self.project_q(q1)
        k1 = self.project_k(k1)

        k1 = rearrange(k1,'b (n dh dw) c -> (b n) (dh dw) c', n=N_, dh=self.window_size, dw=self.window_size)
        
        q1 = rearrange(q1,'b (n dh dw) c -> (b n) c (dh dw)', n=N_, dh=self.window_size, dw=self.window_size)
        agent_tokens = self.pool(q1)

        q1 = rearrange(q1,'(b n) c (dh dw) -> (b n) (dh dw) c', n=N_, dh=self.window_size, dw=self.window_size)
        agent_tokens = rearrange(agent_tokens,'(b n) c p -> (b n) p c', n=N_, p=self.pool_size)
        agent_attn = self.softmax(agent_tokens @ k1.transpose(-2, -1))
        agent_attn = self.attn_drop(agent_attn)

        q_attn = self.softmax(q1 @ agent_tokens.transpose(-2, -1))
        q_attn = self.attn_drop(q_attn).transpose(-2, -1)
        
        y=torch.cat([agent_attn,q_attn],dim=1) #bs,2c,h,w
        y = y.view(N*h*w,-1,self.window_size,self.window_size)

        att=self.attention_embed(y) #bs,c*k*k,h,w
        
        att=att.reshape(N*h*w,self.pool_size,self.kernel_size*self.kernel_size,self.window_size,self.window_size)
        att=att.mean(2,keepdim=False).view(N*h*w,-1,self.pool_size) #bs,c,h*w
        k2=F.softmax(att,dim=-1)
        v1 = rearrange(v1,'(b h w) (dh dw) c -> (b h w) c (dh dw) ', 
                    h=H//self.window_size, w=W//self.window_size, dh=self.window_size, dw=self.window_size)
        v1 = self.pool(v1)
        k2 =  k2 @ v1.transpose(-2,-1)
        k2 =  k2.transpose(-2,-1)

        k1 = rearrange(k1,'(b h w) (dh dw) c-> (b h w) c (dh dw)', h=H//self.window_size, w=W//self.window_size, dh=self.window_size, dw=self.window_size)

        k2 = k2 + k1
        f_attn = rearrange(k2,'(b h w) c (dh dw) -> b (h w) (dh dw c)', 
                     b=N, h=H//self.window_size, w=W//self.window_size, dh=self.window_size, dw=self.window_size)

        if not (self.training or train_mode):
            attn_out = batch_index_fill(v.clone(), f_attn, v2.clone(), idx1, idx2)
        else:
            attn_out = f_attn + v2    #f_attn为hard，v2为simple

        attn_out = rearrange(
            attn_out, 'b (h w) (dh dw c) -> b (c) (h dh) (w dw)', 
            h=H//self.window_size, w=W//self.window_size, dh=self.window_size, dw=self.window_size
        )
        
        out = attn_out
        out = self.act(self.conv_sptial(out))*ca + out  #Convolution Branch
        out = self.project_out(out)

        if self.training:
            return out, torch.mean(mask,dim=1)
        return out



class CAMixer0(nn.Module):
    def __init__(self, dim, window_size=8, bias=True, is_deformable=True, ratio=0.5):
        super().__init__()    

        self.dim = dim
        self.window_size = window_size
        self.is_deformable = is_deformable
        self.ratio = ratio

        k = 3
        d = 2

        self.project_v = nn.Conv2d(dim, dim, 1, 1, 0, bias = bias)
        self.project_q = nn.Linear(dim, dim, bias = bias)
        self.project_k = nn.Linear(dim, dim, bias = bias)

        # Conv
        self.conv_sptial = nn.Sequential(
            nn.Conv2d(dim, dim, k, padding=k//2, groups=dim),
            nn.Conv2d(dim, dim, k, stride=1, padding=((k//2)*d), groups=dim, dilation=d))        
        self.project_out = nn.Conv2d(dim, dim, 1, 1, 0, bias = bias)

        self.act = nn.GELU()
        # Predictor
        self.route = PredictorLG(dim,window_size,ratio=ratio)

    def forward(self,x,condition_global=None, mask=None, train_mode=False):
        N,C,H,W = x.shape

        v = self.project_v(x)

        if self.is_deformable:
            condition_wind = torch.stack(torch.meshgrid(torch.linspace(-1,1,self.window_size),torch.linspace(-1,1,self.window_size)))\
                    .type_as(x).unsqueeze(0).repeat(N, 1, H//self.window_size, W//self.window_size)
            if condition_global is None:
                _condition = torch.cat([v, condition_wind], dim=1)
            else:
                _condition = torch.cat([v, condition_global, condition_wind], dim=1)

        mask, offsets, ca, sa = self.route(_condition,ratio=self.ratio,train_mode=train_mode)

        q = x 
        k = x + flow_warp(x, offsets.permute(0,2,3,1), interp_mode='bilinear', padding_mode='border')
        qk = torch.cat([q,k],dim=1)

        vs = v*sa

        v  = rearrange(v,'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=self.window_size, dw=self.window_size)
        vs = rearrange(vs,'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=self.window_size, dw=self.window_size)
        qk = rearrange(qk,'b c (h dh) (w dw) -> b (h w) (dh dw c)', dh=self.window_size, dw=self.window_size)

        if self.training or train_mode:
            N_ = v.shape[1]
            v1,v2 = v*mask, vs*(1-mask)   
            qk1 = qk*mask 
        else:
            idx1, idx2 = mask
            _, N_ = idx1.shape
            v1,v2 = batch_index_select(v,idx1),batch_index_select(vs,idx2)
            qk1 = batch_index_select(qk,idx1)

        v1 = rearrange(v1,'b n (dh dw c) -> (b n) (dh dw) c', n=N_, dh=self.window_size, dw=self.window_size)
        qk1 = rearrange(qk1,'b n (dh dw c) -> b (n dh dw) c', n=N_, dh=self.window_size, dw=self.window_size)

        q1,k1 = torch.chunk(qk1,2,dim=2)
        q1 = self.project_q(q1)
        k1 = self.project_k(k1)
        q1 = rearrange(q1,'b (n dh dw) c -> (b n) (dh dw) c', n=N_, dh=self.window_size, dw=self.window_size)
        k1 = rearrange(k1,'b (n dh dw) c -> (b n) (dh dw) c', n=N_, dh=self.window_size, dw=self.window_size)
  
        attn = q1 @ k1.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        f_attn = attn@v1
        # print('attn.shape',attn.shape)
        # print('v1.shape',v1.shape)
        
        f_attn = rearrange(f_attn,'(b n) (dh dw) c -> b n (dh dw c)', 
            b=N, n=N_, dh=self.window_size, dw=self.window_size)

        if not (self.training or train_mode):
            attn_out = batch_index_fill(v.clone(), f_attn, v2.clone(), idx1, idx2)
        else:
            attn_out = f_attn + v2    #f_attnw为hard，v2为simple

        attn_out = rearrange(
            attn_out, 'b (h w) (dh dw c) -> b (c) (h dh) (w dw)', 
            h=H//self.window_size, w=W//self.window_size, dh=self.window_size, dw=self.window_size
        )
        
        out = attn_out
        out = self.act(self.conv_sptial(out))*ca + out  #Convolution Branch
        out = self.project_out(out)

        if self.training:
            return out, torch.mean(mask,dim=1)
        return out


    def __init__(self, dim, window_size=8, bias=True, ratio=0.5,attn_drop=0):
        super().__init__()    

        self.dim = dim
        self.window_size = window_size
        # self.ratio = ratio

        # self.attn_drop = nn.Dropout(attn_drop)

        k = 3
        d = 2

        self.project_v = nn.Conv2d(dim, dim, 1, 1, 0, bias = bias)
        self.project_q = nn.Conv2d(dim, dim, 1, 1, 0, bias = bias)
        self.project_k = nn.Conv2d(dim, dim, 1, 1, 0, bias = bias)

        # self.softmax = nn.Softmax(dim=-1)
        

        
    def forward(self,x):
        N,C,H,W = x.shape 
        # print('x.shape',x.shape)

        v1 = self.project_v(x)       

        q1 = self.project_q(x)

        k1 = self.project_k(x)
        
        v1 = rearrange(v1,'b c (h dh) (w dw) -> (b h w) (dh dw) c', 
                       h=H//self.window_size, w=W//self.window_size, dh=self.window_size, dw=self.window_size)
        q1 = rearrange(q1,'b c (h dh) (w dw) -> (b h w) (dh dw) c', 
                       h=H//self.window_size, w=W//self.window_size, dh=self.window_size, dw=self.window_size)
        k1 = rearrange(k1,'b c (h dh) (w dw) -> (b h w) (dh dw) c',
                       h=H//self.window_size, w=W//self.window_size, dh=self.window_size, dw=self.window_size)
        # print('q1.shape',q1.shape)
        # print('k1.shape',k1.shape)

        attn = q1 @ k1.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        # print('attn.shape',attn.shape)
        # print('v1.shape',v1.shape)

        f_attn = attn@v1

        
        f_attn = rearrange(f_attn,'(b h w) (dh dw) c -> b (c) (h dh) (w dw)', 
                     b=N, h=H//self.window_size, w=W//self.window_size, dh=self.window_size, dw=self.window_size)
        return f_attn
    def __init__(self, dim, window_size=8, bias=True, ratio=0.5,agent_num=49,attn_drop=0):
        super().__init__()    

        self.dim = dim
        self.window_size = window_size
        self.ratio = ratio

        self.agent_num = agent_num
        self.attn_drop = nn.Dropout(attn_drop)

        k = 3
        d = 2

        self.project_v = nn.Conv2d(dim, dim, 1, 1, 0, bias = bias)
        self.project_q = nn.Conv2d(dim, dim, 1, 1, 0, bias = bias)
        self.project_k = nn.Conv2d(dim, dim, 1, 1, 0, bias = bias)

        self.softmax = nn.Softmax(dim=-1)
        
        # agent token
        self.pool_size = int(agent_num ** 0.5)
        self.pool = nn.AdaptiveAvgPool1d(self.pool_size)
        
    def forward(self,x):
       N,C,H,W = x.shape 
       v1 = self.project_v(x)       

       q1 = self.project_q(x)

       k1 = self.project_k(x)
       # print('k1.shape',k1.shape)
       
       v1 = rearrange(v1,'b c (h dh) (w dw) -> (b h w) (dh dw) c', 
                      h=H//self.window_size, w=W//self.window_size, dh=self.window_size, dw=self.window_size)
       q1 = rearrange(q1,'b c (h dh) (w dw) -> (b h w) c (dh dw)', 
                      h=H//self.window_size, w=W//self.window_size, dh=self.window_size, dw=self.window_size)
       k1 = rearrange(k1,'b c (h dh) (w dw) -> (b h w) c (dh dw)',
                      h=H//self.window_size, w=W//self.window_size, dh=self.window_size, dw=self.window_size)
       agent_tokens = self.pool(q1)   #尝试取q1和k1的比例和

       k1 = rearrange(k1,'(b h w) c (dh dw) -> (b h w) (dh dw) c',
                      h=H//self.window_size, w=W//self.window_size, dh=self.window_size, dw=self.window_size)
       q1 = rearrange(q1,'(b h w) c (dh dw) -> (b h w) (dh dw) c', 
                      h=H//self.window_size, w=W//self.window_size, dh=self.window_size, dw=self.window_size)
       agent_tokens = rearrange(agent_tokens,'(b h w) c p -> (b h w) p c', 
                                h=H//self.window_size, w=W//self.window_size, p=self.pool_size)
       
       agent_attn = self.softmax(agent_tokens @ k1.transpose(-2, -1))
       agent_attn = self.attn_drop(agent_attn)


       agent_v = agent_attn @ v1

       q_attn = self.softmax(q1 @ agent_tokens.transpose(-2, -1))
       q_attn = self.attn_drop(q_attn)
       
       f_attn = q_attn @ agent_v



       f_attn = rearrange(f_attn,'(b h w) (dh dw) c -> b (c) (h dh) (w dw)', 
                    b=N, h=H//self.window_size, w=W//self.window_size, dh=self.window_size, dw=self.window_size)
       return f_attn
   
class FFTCA(nn.Module):
    def __init__(self, dim, window_size=8, bias=True, ratio=0.5,agent_num=49,attn_drop=0):
        super().__init__()    

        self.dim = dim
        self.window_size = window_size
        self.ratio = ratio

        self.agent_num = agent_num
        self.attn_drop = nn.Dropout(attn_drop)
        self.kernel_size = 3
        k = 3
        d = 2

        self.project_v = nn.Conv2d(dim, dim, 1, 1, 0, bias = bias)
        self.project_k = nn.Conv2d(dim, dim, 1, 1, 0, bias = bias)

        self.softmax = nn.Softmax(dim=-1)
        
        # agent token
        self.pool_size = int(agent_num ** 0.5)
        self.pool = nn.AdaptiveAvgPool1d(self.pool_size)
        factor=2
        self.attention_embed=nn.Sequential(
            nn.Conv2d(2*self.pool_size,2*self.pool_size//factor,1,bias=False),
            nn.BatchNorm2d(2*self.pool_size//factor),
            nn.ReLU(),
            nn.Conv2d(2*self.pool_size//factor,self.kernel_size*self.kernel_size*self.pool_size,1)
        )
                
    def forward(self,x):
       N,C,H,W = x.shape 
       h=H//self.window_size
       w=W//self.window_size
       v1 = self.project_v(x)       

       k1 = self.project_k(x)

       v1 = rearrange(v1,'b c (h dh) (w dw) -> (b h w) c (dh dw) ', 
                      h=H//self.window_size, w=W//self.window_size, dh=self.window_size, dw=self.window_size)
       q1 = rearrange(x,'b c (h dh) (w dw) -> (b h w) c (dh dw)', 
                      h=H//self.window_size, w=W//self.window_size, dh=self.window_size, dw=self.window_size)
       k1 = rearrange(k1,'b c (h dh) (w dw) -> (b h w) c (dh dw)',
                      h=H//self.window_size, w=W//self.window_size, dh=self.window_size, dw=self.window_size)
       agent_tokens = self.pool(q1)   #尝试取q1和k1的比例和

       v1 = self.pool(v1)

       k1 = rearrange(k1,'(b h w) c (dh dw) -> (b h w) (dh dw) c',
                      h=H//self.window_size, w=W//self.window_size, dh=self.window_size, dw=self.window_size)
       q1 = rearrange(q1,'(b h w) c (dh dw) -> (b h w) (dh dw) c', 
                      h=H//self.window_size, w=W//self.window_size, dh=self.window_size, dw=self.window_size)
       agent_tokens = rearrange(agent_tokens,'(b h w) c p -> (b h w) p c', 
                                h=H//self.window_size, w=W//self.window_size, p=self.pool_size)
       
       agent_attn = self.softmax(agent_tokens @ k1.transpose(-2, -1))
       agent_attn = self.attn_drop(agent_attn)
       
       q_attn = self.softmax(q1 @ agent_tokens.transpose(-2, -1))
       q_attn = self.attn_drop(q_attn).transpose(-2, -1)
       y=torch.cat([agent_attn,q_attn],dim=1) #bs,2c,h,w
       # print('y.shape',y.shape)

       y = y.view(N*h*w,-1,self.window_size,self.window_size)
       att=self.attention_embed(y) #bs,c*k*k,h,w
       att=att.reshape(N*h*w,self.pool_size,self.kernel_size*self.kernel_size,self.window_size,self.window_size)
       att=att.mean(2,keepdim=False).view(N*h*w,-1,self.pool_size) #bs,c,h*w
       k2=F.softmax(att,dim=-1)
       k2 =  k2 @ v1.transpose(-2,-1)
       k2 =  k2.transpose(-2,-1)
       k1 = rearrange(k1,'(b h w) (dh dw) c-> (b h w) c (dh dw)', h=H//self.window_size, w=W//self.window_size, dh=self.window_size, dw=self.window_size)
       k2 = k2 + k1
       k2 = rearrange(k2,'(b h w) c (dh dw) -> b (c) (h dh) (w dw)', 
                    b=N, h=H//self.window_size, w=W//self.window_size, dh=self.window_size, dw=self.window_size)
       
       
       return k2

class MDFM(nn.Module):
    def __init__(self, dim, mult = 1, bias=False, dropout = 0.):
        super().__init__()
        self.dim = dim

        self.project_in = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.dwconv3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.dwconv5 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim, bias=bias)

        self.dwconv7 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3, groups=dim, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        
        x1, x2, x3 = x.chunk(3, dim=1)
        x1 = self.dwconv3(x1)
        x2 = self.dwconv5(x2)
        x3 = self.dwconv7(x3)
        # x1, x2, x3 = self.dwconv(x).chunk(3, dim=1)
        x = F.gelu(x1) * x2
        x = F.gelu(x) * x3  #double gated
        x = self.project_out(x)
        return x
    
    def calculate_flops(self,x):
        _,_,H,W = x.shape
        flops = np.longlong(0)
        flops += H*W*self.dim*2*self.dim*2
        flops += H*W*2*self.dim*9*2
        flops += H*W*self.dim*self.dim*2
        return flops


class CADBlock(nn.Module):
    def __init__(self, n_feats, window_size=8, ratio=0.5):
        super(CADBlock,self).__init__()
        
        self.n_feats = n_feats
        self.norm1 = LayerNorm(n_feats)
        self.mixer = CAMM(n_feats,window_size=window_size,ratio=ratio)
        self.aat = FFTCA(n_feats,window_size=window_size,ratio=ratio)
        self.norm2 = LayerNorm(n_feats)
        self.ffn = MDFM(n_feats)

        
    def forward(self,x,condition_global=None,attention=None):
        if self.training:
            res, decision = self.mixer(x,condition_global)
            res = self.aat(res)
            x = self.norm1(x+res)
            res = self.ffn(x)
            x = self.norm2(x+res)
            x = x*attention
            
            return x, decision
        else:
            res = self.mixer(x,condition_global)
            res = self.aat(res)
            x = self.norm1(x+res)
            res = self.ffn(x)
            x = self.norm2(x+res)
            x = x*attention

            return x 
    

class CADGroup(nn.Module):
    def __init__(self, n_feats, n_block, window_size=8, ratio=0.5):
        super(CADGroup, self).__init__()
        
        self.n_feats = n_feats

        self.body = nn.ModuleList([CADBlock(n_feats, window_size=window_size, ratio=ratio) for i in range(n_block)])
        self.body_tail = nn.Conv2d(n_feats, n_feats, 1, 1, 0)
        
    def forward(self,x,condition_global=None,attention=None):
        decision = []
        shortcut = x.clone()
        if self.training:
            for _, blk in enumerate(self.body):
                x, mask = blk(x,condition_global,attention)
                decision.append(mask)
            x = self.body_tail(x) + shortcut
            return x, decision
        else:
            for _, blk in enumerate(self.body):
                x = blk(x,condition_global,attention)
            x = self.body_tail(x) + shortcut
            return x 

    def calculate_flops(self,x):
        _,_,H,W = x.shape
        flops = np.longlong(0)
        flops += H*W*self.n_feats*self.n_feats*2
        return flops

class Feature_Refinement_Block(nn.Module):  
    def __init__(self, channel, reduction):
        super(Feature_Refinement_Block, self).__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.sa = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.Conv2d(channel, channel // 8, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 3, 1, 1),
            nn.Sigmoid()
        )

        self.s_embed=nn.Sequential(
            nn.Conv2d(channel,channel,3,padding=3//2,groups=4,bias=False),
            nn.ReLU(inplace=True)
        )
        self.project = nn.Conv2d(channel, channel, 3, 1, 1)
        
    def forward(self, x):
        a = self.ca(x)
        t = self.sa(x)
        s = torch.mul(a, x) + torch.mul(t, x)
        s = self.s_embed(s) + self.project(x)
        return s

        
#@ARCH_REGISTRY.register()
class CADNetwork(nn.Module):
    def __init__(self, n_block=[4,4,6,6], n_group=4, n_colors=3, n_feats=60, scale=4, ratio=0.5, tile=None):
        super().__init__()
        self.n_feats = n_feats
        self.window_sizes = 16
        self.tile = tile
        self.ratio=ratio

        self.head = nn.Conv2d(n_colors, n_feats, 3, 1, 1)

        self.global_predictor = nn.Sequential(nn.Conv2d(n_feats, 8, 1, 1, 0, bias=True),
                                        nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                        nn.Conv2d(8, 2, 3, 1, 1, bias=True),
                                        nn.LeakyReLU(negative_slope=0.1, inplace=True))

        self.scale = scale
        # define body module
        self.body = nn.ModuleList([CADGroup(n_feats, n_block=n_block[i], window_size=self.window_sizes, ratio=ratio) for i in range(n_group)])
        self.body_tail = nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        # define tail module
        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, n_colors*(scale**2), 3, 1, 1),
            nn.PixelShuffle(scale)
        )
        
        self.FTCA= FTCA(dim=3)

        self.test = nn.Conv2d(3,1,3,1,1)
        
        self.FRB = Feature_Refinement_Block(n_feats, reduction=8)

    def forward_origin(self, x):
        decision = []
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        # rain streaks attention map
        attention = self.FTCA(x)
        attention = self.test(attention)                 

        x = self.head(x)
        x = self.FRB(x)
        condition_global = self.global_predictor(x)
        shortcut = x.clone()
        if self.training:
            for _, blk in enumerate(self.body):
                x, mask = blk(x,condition_global,attention)
                decision.extend(mask)
        else:
            for _, blk in enumerate(self.body):
                x = blk(x,condition_global,attention)  
                 
        
        x = self.body_tail(x) 
        x = x + shortcut
        x = self.tail(x)

        if self.training:
            return attention, x[:, :, 0:H*self.scale, 0:W*self.scale],  2*self.ratio*(torch.mean(torch.cat(decision,dim=1),dim=(0,1))-0.5)**2 
            # print("self.training1", self.training)
        else:
            return attention, x[:, :, 0:H*self.scale, 0:W*self.scale], 1
            # print("self.training2", self.training)
    def forward(self, img_lq,tile=None):
        tile = self.tile
        # print("self.training3", self.training)
        if tile is None or self.training:
            # test the image as a whole
            attention,output,des = self.forward_origin(img_lq)
        else:
            # test the image tile by tile or use TileModel
            b, c, h, w = img_lq.size()
            tile = min(tile, h, w)
            tile_overlap = tile//16
            sf = self.scale

            stride = tile - tile_overlap
            h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
            w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
            E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)
            W = torch.zeros_like(E)

            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                    # gt_patch = img_gt[..., h_idx:h_idx+tile, w_idx:w_idx+tile]

                    attention,out_patch,des = self.forward_origin(in_patch)
                    out_patch_mask = torch.ones_like(out_patch)

                    E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                    W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
            output = E.div_(W)

        return attention,output,des

    def check_image_size(self, x, ):
        _, _, h, w = x.size()
        wsize = self.window_sizes
        # for i in range(1, len(self.window_sizes)):
        #     wsize = wsize*self.window_sizes[i] // math.gcd(wsize, self.window_sizes[i])
        mod_pad_h = (wsize - h % wsize) % wsize
        mod_pad_w = (wsize - w % wsize) % wsize
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def flops(self,x):
        _,_,H,W = x.shape
        flops = np.longlong(0)
        #head
        flops += np.longlong(H*W*3*self.n_feats*9*2)
        #global predictor
        flops += np.longlong(H*W*8*(self.n_feats+2)*9*2)
        #body
        for layer in self.modules():
            if hasattr(layer, 'calculate_flops'):
                temp = layer.calculate_flops(x)
                flops += np.longlong(temp)
        #body tail
        flops += np.longlong(H*W*self.n_feats*self.n_feats*9*2)
        #tail
        flops += np.longlong(H*W*self.n_feats*(self.scale**2)*3*9*2)
        return flops





if __name__ == '__main__':

    x1 = torch.randn((1,3,64,64))
    x2 = torch.randn((1,3,32,32))

    net = CADNetwork(scale=1, n_feats=60, ratio=0.5)
    # net.eval()
    net.train()
    num_parameters = sum(map(lambda x: x.numel(), net.parameters()))
    print('{:>16s} : {:<.4f} [M]'.format('#Params', num_parameters/10**6))
    print('{:>16s} : {:<.4f} [G]'.format('#FLOPs', net.flops(x1)/10**9))
    print(net(x1)[0].shape, net(x1)[1].shape)


    # macs,params=profile(net,inputs=(x1,),verbose=True)
    # flops=2*macs
    # MACs,FLOPs,Params=thop.clever_format([macs,flops,params],'%.3f')
    
    # print('flops: ',flops/10**9,'[G]')






        