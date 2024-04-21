import copy
import math
from turtle import pos

import numpy as np
import torch
import torch.nn as nn

"""
code from https://github.com/harvardnlp/annotated-transformer/blob/master/the_annotated_transformer.py
"""

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9) 
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask = None, use_att = True): 
        "Implements Figure 2"
        if use_att: 
            if mask is not None:
                
                mask = mask.unsqueeze(1)
                mask = mask.unsqueeze(-2)  

            nbatches = query.size(0)

            
            query, key, value = [
                lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                for lin, x in zip(self.linears, (query, key, value))
            ]

            
            x, self.attn = attention(
                query, key, value, mask=mask, dropout=self.dropout
            )

            
            x = (
                x.transpose(1, 2)
                .contiguous()
                .view(nbatches, -1, self.h * self.d_k)
            )
            del query
            del key
            del value
            return self.linears[-1](x)
        
        else: 
            query = self.linears[0](query)
            query = self.linears[-1](query)
            return query

    
    def get_mask(self, event_list, bs, num_cls):
        mask = torch.zeros([bs, num_cls])
        mask[[event_list[0], event_list[1]]]=1  
        return mask
            

class PositionalEncoding(nn.Module):
    """Implement the PE function.
    
    """

    def __init__(self, 
                d_model, 
                dropout, 
                max_len=35 
                ):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.max_len = max_len

        
        
    def forward(self, x, pe): 
        x = x + pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

    def get_position_encoding(self, bs, position):
        pe = torch.zeros(bs, self.max_len, self.d_model)
        if position == None:  
            position = torch.arange(0, self.max_len).unsqueeze(1) 
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2) * -(math.log(10000.0) / self.d_model)
        )

        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        pe=pe.to('cuda')
        
        return pe

    def get_position(self, event_list, bs, num_cls):  
        position = torch.zeros(bs, num_cls)
        pos_tensor = np.array(event_list[2]).astype(np.float32)  
        pos_tensor = torch.from_numpy(pos_tensor)
        position[[event_list[0], event_list[1]]] = pos_tensor
        position = position.unsqueeze(2)  
        return position


class MHSALayer(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    

    def __init__(self, size, dropout):
        super(MHSALayer, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class LayerNorm(nn.Module):
    "Construct a layernorm module in the OpenAI style (epsilon inside the square root)."

    def __init__(self, n_state, e=1e-5):
        super(LayerNorm, self).__init__()
        self.g = nn.Parameter(torch.ones(n_state))
        self.b = nn.Parameter(torch.zeros(n_state))
        self.e = e

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.e)
        return self.g * x + self.b