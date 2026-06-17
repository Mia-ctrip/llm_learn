import re
import torch
import math
import numpy as np
import torch.nn as nn

def self_attention(q, k, v, mask=None)-> torch.Tensor:
    #公式 softmax(qk^T/d_k)·v
    d_model = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_model)
    if mask is not None:
        #实现掩码 给q @ k.transpose(-2, -1)中某个词后续的序列都设置为-∞
        scores = scores.masked_fill(mask == 0, -1e9) 
    return torch.softmax(scores, dim=-1) @ v


def multi_head_attention(q, k, v, h, mask=None)-> torch.Tensor:
   # 多头注意力机制 qkv维度扩展了h倍
    d_model = q.size(-1)
    q = q.view(q.size(0), q.size(1), h, -1).transpose(1, 2)
    k = k.view(k.size(0), k.size(1), h, -1).transpose(1, 2)
    v = v.view(v.size(0), v.size(1), h, -1).transpose(1, 2)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_model)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9) 
    return torch.softmax(scores, dim=-1) @ v

def ffn(x) -> torch.Tensor:
    # 前馈神经网络
    return nn.Sequential(
        nn.Linear(x.size(-1), x.size(-1)*4), 
        nn.ReLU(), 
        nn.Linear(x.size(-1)*4, x.size(-1))
    )(x)


def position_embedding(input_embedding)-> torch.Tensor:
    # 位置编码
    # 只涉及计算 不用训练 
    pe = torch.zeros(input_embedding.size(0), input_embedding.size(1), input_embedding.size(2))
    return pe + input_embedding

def linear(x) -> torch.Tensor:
    return nn.Linear(x.size(-1), x.size(-1))(x)


class EncodeLayer(nn.Module):
    def __init__(self, input_embedding):
        self.wq = torch.randn(input_embedding.size(-1), input_embedding.size(-1))
        self.wk = torch.randn(input_embedding.size(-1), input_embedding.size(-1))
        self.wv = torch.randn(input_embedding.size(-1), input_embedding.size(-1))
        super(EncodeLayer, self).__init__()
    
    def forward(self, input_embedding):
        #编码层    
        #先计算q k v矩阵
        q = self.wq @ input_embedding
        k = self.wk @ input_embedding
        v = self.wv @ input_embedding
        z = self_attention(q, k, v, None)
        #再计算残差 torch怎么normalize？
        residual = torch.layer_norm(input_embedding + z)
        #残差送入FNN层
        fnn_output = ffn(residual)
        r = torch.layer_norm(fnn_output + residual)
        return r

class DecodeLayer(nn.Module):        
    def __init__(self, input_embedding):
        self.wq = torch.randn(input_embedding.size(-1), input_embedding.size(-1))
        self.wk = torch.randn(input_embedding.size(-1), input_embedding.size(-1))
        self.wv = torch.randn(input_embedding.size(-1), input_embedding.size(-1))
        super(DecodeLayer, self).__init__()
    
    def forward(self, input_embedding):
        #先计算q k v矩阵
        q = self.wq @ input_embedding
        k = self.wk @ input_embedding
        v = self.wv @ input_embedding
        z = self_attention(q, k, v, True)
        #再计算残差 torch怎么normalize？
        residual = torch.layer_norm(input_embedding + z)
        #残差送入FNN层
        fnn_output = ffn(residual)
        r = torch.layer_norm(fnn_output + residual)
        return r

class GPT(nn.Module):
    def __init__(self, input_embedding):
        super(GPT, self).__init__()
        self.position_embedding = position_embedding(input_embedding)
        self.decode_layer = DecodeLayer(input_embedding)

    def forward(self, input_embedding):
        z1 = self.decode_layer(input_embedding)
        z2 = self.decode_layer(z1)
        z3 = self.decode_layer(z2)
        r = linear(z3)
        return torch.softmax(r, dim=-1)


if __name__ == '__main__':
    input_embedding = torch.randn(1, 10, 20)
    model = GPT(input_embedding)
    output = model(input_embedding)
    print(output.shape)
