import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy


class CrossGATLayer(nn.Module):


    def __init__(self, in_features, out_features, dropout=0, alpha=0.1, concat=True, hybrid_cond=0., t_ksize=1, s_ksize=4):
        super(CrossGATLayer, self).__init__()

        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.hybrid_cond = hybrid_cond
        self.t_ksize=t_ksize
        self.s_ksize=s_ksize
        assert max(self.t_ksize,self.s_ksize)>=1

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
        self.attention=attention 

        h_prime = torch.matmul(attention, Wh)

        h_prime = F.elu(h_prime) 
        x_a, x_v= torch.chunk(h_prime, chunks=2, dim=1)
        return x_a, x_v

    def _prepare_attentional_mechanism_input(self, Wh):
        

        
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :]) 
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])

        e = Wh1 + Wh2.transpose(-1, -2) 
        return self.leakyrelu(e)

    def _get_semantic_adj(self, x, y=None, k=10, dis='euc'): 
        """
        https://github.com/frostinassiky/gtad/blob/master/gtad_lib/models.py

        """
        def euclidean_distance(x,y):
            inner = 2 * torch.matmul(x,y.transpose(-2, -1)) 
            xx = torch.sum(x ** 2, dim=-1, keepdim=True)
            yy = torch.sum(y ** 2, dim=-1, keepdim=True)
            pairwise_distance = -(xx - inner + yy.transpose(-2, -1)) 
            
            return pairwise_distance
        
        def cosine_distance(x,y):
            x_norm=torch.norm(x,dim=-1,keepdim=True)
            y_norm=torch.norm(y,dim=-1,keepdim=True)
            xy_norm=x_norm*y_norm.transpose(-2,-1)
            xy_dot=torch.matmul(x,y.transpose(-2, -1))
            pairwise_distance=xy_dot/xy_norm
            return pairwise_distance

        assert len(x.shape)==3 
        if y is None:
            y = x

        bs, t, _=x.shape
        assert t%2==0
        t=t//2 
        sem_adj=torch.zeros(bs,2*t,2*t).to('cuda') 

        assert dis in ['euc','cos']
        if dis == 'euc':
            pairwise_distance=euclidean_distance(x,y)
        elif dis=='cos':
            pairwise_distance=cosine_distance(x,y)
        else:
            raise NotImplementedError('not implenment distance')

        _, idx_aa = pairwise_distance[:,:t,:t].topk(k=k, dim=-1)  
        sem_adj[:,:t,:t].scatter_(-1,torch.cuda.LongTensor(idx_aa),1) 

        _, idx_vv = pairwise_distance[:,t:,t:].topk(k=k, dim=-1)
        sem_adj[:,t:,t:].scatter_(-1,torch.cuda.LongTensor(idx_vv),1)
        
        return sem_adj

    def _get_temporal_adj(self, x, k=3):
        """
        x: feature (b,t,dim)
        return tem_adj: (1, t, t)
        """
        _,t,_=x.shape
        assert t%2==0 
        t=t//2 

        assert k%2==1 
        k=k//2 
        tem_adj=np.zeros([2*t,2*t])
        for i in range(2*t): 
            for j in range(2*t):
                if abs(i-j)-0.5<=k and (i-(t-0.5))*(j-(t-0.5))>0: 
                    tem_adj[i][j]=1
                
                
        tem_adj=torch.tensor(tem_adj).to('cuda') 
        tem_adj=tem_adj.unsqueeze(0) 
        
        return tem_adj
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class DyGAT(nn.Module):
    def __init__(self, input_dim, model_dim=None, output_dim=None, dropout=0, alpha=0.1, num_heads=1, residual=False):
        """Dense version of GAT."""
        super(DyGAT, self).__init__()

        model_dim = model_dim or input_dim
        output_dim = output_dim or input_dim
        
        assert num_heads==1 

        self.dropout = dropout
        self.residual=residual
        self.attention = CrossGATLayer(input_dim,
                                               model_dim,
                                               dropout=dropout,
                                               alpha=alpha,
                                               concat=True)
        
        

        self.out_att = CrossGATLayer(model_dim * num_heads,
                                           output_dim,
                                           dropout=dropout,
                                           alpha=alpha,
                                           concat=False)
        
        self.att1=[]
        self.att2=[]

    def forward(self, x_a, x_v, adj): 
        self.att1=[]
        self.att2=[]
        
        if self.residual: 
            x_a_in=x_a
            x_v_in=x_v

        num_classes=adj.shape[0]
        x_a = F.dropout(x_a, self.dropout, training=self.training)
        x_v = F.dropout(x_v, self.dropout, training=self.training)     
        x_a_layer1, x_v_layer1=self.attention(x_a, x_v, adj[0]) 
        self.att1.append(self.attention.attention)
        
        for i in range(1,num_classes): 
            x_a_new, x_v_new=self.attention(x_a, x_v, adj[i])
            x_a_layer1=x_a_layer1+x_a_new
            x_v_layer1=x_v_layer1+x_v_new
            self.att1.append(self.attention.attention)
        
        x_a_layer1=x_a_layer1/num_classes
        x_v_layer1=x_v_layer1/num_classes

        x_a_layer1 = F.dropout(x_a_layer1, self.dropout, training=self.training)
        x_v_layer1 = F.dropout(x_v_layer1, self.dropout, training=self.training)

        x_a_out, x_v_out=self.out_att(x_a_layer1, x_v_layer1, adj[0]) 
        self.att2.append(self.out_att.attention)
        
        for i in range(1,num_classes): 
            x_a_new, x_v_new=self.out_att(x_a_layer1, x_v_layer1, adj[i])
            x_a_out=x_a_out+x_a_new
            x_v_out=x_v_out+x_v_new
            self.att2.append(self.out_att.attention)    
        
        x_a_out=x_a_out/num_classes
        x_v_out=x_v_out/num_classes

        if self.residual:
            x_a_out+=x_a_in
            x_v_out+=x_v_in

        return x_a_out, x_v_out


