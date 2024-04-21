

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy


class CrossGATLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout=0, alpha=0.1, act=True, hybrid_cond=0.):
        super(CrossGATLayer, self).__init__()

        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.act = act
        self.hybrid_cond = hybrid_cond

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features), device='cuda'))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x_a, x_v, adj=None): 
        
        bs,t,_=x_a.shape
        h=torch.cat([x_a, x_v],dim=1)
        Wh = torch.matmul(h, self.W)  
        
        e = self._prepare_attentional_mechanism_input(Wh) 
        zero_vec = -9e15 * torch.ones_like(e) 
        if adj is None:
            adj=torch.zeros(bs,2*t,2*t).to('cuda') 
            if self.t_ksize>=1: 
                tem_adj=self._get_temporal_adj(Wh, k=self.t_ksize)
                adj=adj+tem_adj
            if self.s_ksize>=1: 
                sem_adj=self._get_semantic_adj(Wh, k=self.s_ksize)
                adj=adj+sem_adj
        self.adj=adj
        
        

        if self.hybrid_cond == 0: 
            attention = torch.where(adj > 0, e, zero_vec)  
        else: 
            raise NotImplementedError('will update in the future')
        
        
        attention = F.softmax(attention, dim=-1) 

        
        attention = F.dropout(attention, self.dropout, training=self.training) 
        
        h_prime = torch.matmul(attention, Wh)
        
        
        if self.act:
            h_prime = F.elu(h_prime) 
        x_a, x_v= torch.chunk(h_prime, chunks=2, dim=1)
        return x_a, x_v

    def _prepare_attentional_mechanism_input(self, Wh):

        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :]) 
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])

        e = Wh1 + Wh2.transpose(-1, -2) 
        return self.leakyrelu(e)
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SingleGAT(nn.Module):
    def __init__(self, input_dim, model_dim=None, output_dim=None, dropout=0, alpha=0.1, num_heads=1, depth = 2):
        """Dense version of GAT."""
        super(SingleGAT, self).__init__()

        model_dim = model_dim or input_dim
        output_dim = output_dim or input_dim
        
        assert depth >= 1
        assert num_heads >= 1
        

        self.dropout = dropout
        
        
        self.depth = depth
        self.num_heads = num_heads
        self.attentions = []
        for i in range(depth-1):
            if i ==0:
                layer_input = input_dim
            else:
                layer_input = model_dim * num_heads
            self.attentions.append([CrossGATLayer(layer_input,
                                               model_dim,
                                               dropout=dropout,
                                               alpha=alpha,
                                               act=True) for _ in range(num_heads)])

        for i, attention_layer in enumerate(self.attentions): 
            for j, attention in enumerate(attention_layer):    
                self.add_module('attention_{}_{}'.format(i,j), attention)
        
        if depth == 1: 
            outlayer_input = input_dim
        else:
            outlayer_input = model_dim * num_heads
        self.out_att = [CrossGATLayer(outlayer_input,
                                        output_dim,
                                        dropout=dropout,
                                        alpha=alpha,
                                        act=False) for _ in range(num_heads)] 
        for i, attention in enumerate(self.out_att):    
                self.add_module('out_att_{}'.format(i), attention)


    def forward(self, x_a, x_v, adj): 

        for i in range(self.depth-1):
            x_a = F.dropout(x_a, self.dropout, training=self.training)
            x_v = F.dropout(x_v, self.dropout, training=self.training)

            a_list=[]
            v_list=[]
            for att in self.attentions[i]:
                x_a_new, x_v_new=att(x_a, x_v,adj)
                a_list.append(x_a_new)
                v_list.append(x_v_new)

            x_a=torch.cat(a_list,dim=-1)
            x_v=torch.cat(v_list,dim=-1)

        x_a = F.dropout(x_a, self.dropout, training=self.training)
        x_v = F.dropout(x_v, self.dropout, training=self.training)
        a_list=[]
        v_list=[]
        for att in self.out_att:
            x_a_new, x_v_new = att(x_a, x_v, adj)
            a_list.append(x_a_new)
            v_list.append(x_v_new)
        
        for i in range(1, self.num_heads):
            a_list[0] = a_list[0] + a_list[i]
            v_list[0] = v_list[0] + v_list[i]
        x_a = a_list[0] / self.num_heads  
        x_v = v_list[0] / self.num_heads
        
        x_a = F.elu(x_a)  
        x_v = F.elu(x_v)
        
        return x_a, x_v


