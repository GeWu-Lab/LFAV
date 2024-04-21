"""
Network in stage three. It is based on graph attention network to achieve feature aggregation in event.
aims of stage3 is to interactive
"""

from turtle import forward
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from .graph_modules_single import SingleGAT
from .mhsa_layer import PositionalEncoding, MultiHeadedAttention, MHSALayer, LayerNorm

from tools.distance import cosine_distance


class EventInteractionNet(nn.Module):
    
    def __init__(self,
                 args,
                 model_dim=512,
                 event_interaction_op='attn',
                 num_classes=35):
        super(EventInteractionNet, self).__init__()

        self.num_cls = num_classes
        self.args = args
        self.event_interaction_op = event_interaction_op
        if args.num_stages == 3:
            if args.event_interaction_op == 'attn':
                assert args.s3_within_modal | args.s3_cross_modal
            if args.event_interaction_op == 'mhsa':
                pass
                
                

        if event_interaction_op == 'attn': 
            self.gat = SingleGAT(input_dim=model_dim, 
                                num_heads=args.s3_gat_nheads,
                                depth = args.s3_gat_depth,
                                dropout = args.s3_dropout 
                                ) 
        
        elif event_interaction_op == 'mhsa':  
            self.pe = PositionalEncoding(d_model=model_dim,
                                            dropout=args.s3_dropout, 
                                            max_len=35) 
            
            if args.s3_no_share_weight: 

                self.mhsa_a = MultiHeadedAttention(h=args.s3_mhsa_nheads,
                                                d_model=model_dim,
                                                dropout=args.s3_dropout)
                self.mhsa_v = MultiHeadedAttention(h=args.s3_mhsa_nheads,
                                                d_model=model_dim,
                                                dropout=args.s3_dropout)

                self.mhsa_av = MultiHeadedAttention(h=args.s3_mhsa_nheads,  
                                                d_model=model_dim,
                                                dropout=args.s3_dropout)
                self.mhsa_va = MultiHeadedAttention(h=args.s3_mhsa_nheads,  
                                                d_model=model_dim,
                                                dropout=args.s3_dropout)
                
                self.norm_a = LayerNorm(n_state=model_dim)
                self.norm_v = LayerNorm(n_state=model_dim)
            
            else:
                self.mhsa = MultiHeadedAttention(h=args.s3_mhsa_nheads,
                                                d_model=model_dim,
                                                dropout=args.s3_dropout)

                self.cm_mhsa = MultiHeadedAttention(h=args.s3_mhsa_nheads,  
                                                d_model=model_dim,
                                                dropout=args.s3_dropout)
                
                self.norm = LayerNorm(n_state=model_dim)
            

        elif event_interaction_op == 'none':
            pass

        else:
            raise NotImplementedError("more models for event cross maybe update in future")

        
        self.dropout = args.s3_dropout
        self.fc_prob = nn.Linear(model_dim, self.num_cls)   
        self.re_cal_prob = ReCalProb(model_dim=model_dim, 
                                    feature_detach=args.s3_feature_detach,
                                    event_proj=args.s3_event_proj)
        
    def forward(self, a_event, v_event, a_event_list, v_event_list, a_prob, v_prob, frame_prob, x_a, x_v,  fc_prob = None):
        
        
        bs, num_cls, num_dim = a_event.shape
        
        self.a_event_old = deepcopy(a_event.detach()) 
        self.v_event_old = deepcopy(v_event.detach())

        self.a_event_list = a_event_list[1] 
        self.v_event_list = v_event_list[1]
        
        
        if self.event_interaction_op == 'attn': 
            
            
        

            adj=np.zeros([bs, 2*self.num_cls, 2*self.num_cls]) 
            
            if self.args.s3_within_modal: 
                adj[:, :self.num_cls, :self.num_cls] = self.event_adj(bs, self.num_cls, a_event_list) 
                adj[:, self.num_cls:, self.num_cls:] = self.event_adj(bs, self.num_cls, v_event_list)
            if self.args.s3_cross_modal: 
                adj[:,:self.num_cls, self.num_cls:] = self.cross_modal_event_adj(bs, self.num_cls, a_event_list, v_event_list) 
                adj[:,self.num_cls:, :self.num_cls] = self.cross_modal_event_adj(bs, self.num_cls, v_event_list, a_event_list) 
            
            adj=torch.tensor(adj).to('cuda') 
            
            

            a_event_new, v_event_new = self.gat(a_event, v_event, adj) 
            
            if self.args.s3_residual:
                a_event += a_event_new
                v_event += v_event_new
                
            else:
                a_event = a_event_new
                v_event = v_event_new

        elif self.event_interaction_op =='mhsa':  
            if self.args.s3_attn == 'all':
                s_att = True
                c_att = True
            elif self.args.s3_attn == 'cm':
                s_att = False
                c_att = True
            elif self.args.s3_attn == 'self':
                s_att = True
                c_att = False
            elif self.args.s3_attn == 'none':
                s_att = False
                c_att = False
            else:
                raise NotImplementedError
            
            if self.args.s3_mhsa_pe:  
                a_position = self.pe.get_position(a_event_list, bs, num_cls)
                a_pe = self.pe.get_position_encoding(bs, a_position)
                a_event = self.pe(a_event, a_pe)

                v_position = self.pe.get_position(v_event_list, bs, num_cls)
                v_pe = self.pe.get_position_encoding(bs, v_position)
                v_event = self.pe(v_event, v_pe)
            
            if self.args.s3_cross_modal:
                if self.args.s3_cm_method == 'concat':  
                    a_mask = self.mhsa.get_mask(a_event_list, bs, num_cls)
                    v_mask = self.mhsa.get_mask(v_event_list, bs, num_cls)
                    mask = torch.cat([a_mask, v_mask], dim=1)  
                    mask = mask.to('cuda')
                    
                    event = torch.cat([a_event, v_event], dim=1)  
                    shortcut = event
                    
                    if self.args.s3_pre_norm: 
                        event = self.norm(event)  

                    event = self.mhsa(event, event, event, mask)
                    event = F.dropout(event, self.dropout, training=self.training)
                    event = event + shortcut

                    if self.args.s3_post_norm: 
                        event = self.norm(event)  
                    
                    a_event, v_event = torch.chunk(event, chunks=2, dim=1)
                
                elif self.args.s3_cm_method == 'add':   
                    if not self.args.s3_no_share_weight:  
                        a_mask = self.mhsa.get_mask(a_event_list, bs, num_cls)
                        a_mask = a_mask.to('cuda')
                        v_mask = self.mhsa.get_mask(v_event_list, bs, num_cls)
                        v_mask = v_mask.to('cuda')
                        
                        a_shortcut = a_event
                        v_shortcut = v_event

                        if self.args.s3_pre_norm:
                            a_event = self.norm(a_event)
                            v_event = self.norm(v_event)

                        if self.args.s3_share_cm:
                            a_event_new = self.mhsa(a_event, a_event, a_event, a_mask) + self.mhsa(a_event, v_event, v_event, v_mask)
                            v_event_new = self.mhsa(v_event, v_event, v_event, v_mask) + self.mhsa(v_event, a_event, a_event, a_mask) 
                        elif self.args.s3_just_cm:
                            a_event_new = self.cm_mhsa(a_event, v_event, v_event, v_mask)
                            v_event_new = self.cm_mhsa(v_event, a_event, a_event, a_mask) 
                        else:
                            a_event_new = self.mhsa(a_event, a_event, a_event, a_mask, use_att = s_att) + self.cm_mhsa(a_event, v_event, v_event, v_mask, use_att = c_att)
                            v_event_new = self.mhsa(v_event, v_event, v_event, v_mask, use_att = s_att) + self.cm_mhsa(v_event, a_event, a_event, a_mask, use_att = c_att) 
                            
                            
                        a_event_new = F.dropout(a_event_new, self.dropout, training=self.training)
                        v_event_new = F.dropout(v_event_new, self.dropout, training=self.training)

                        a_event = a_event_new + a_shortcut
                        v_event = v_event_new + v_shortcut

                        if self.args.s3_post_norm:
                            a_event = self.norm(a_event)
                            v_event = self.norm(v_event)
                    
                    else:  
                        
                        a_mask = self.mhsa_a.get_mask(a_event_list, bs, num_cls)
                        a_mask = a_mask.to('cuda')
                        v_mask = self.mhsa_v.get_mask(v_event_list, bs, num_cls)
                        v_mask = v_mask.to('cuda')
                        a_shortcut = a_event
                        v_shortcut = v_event

                        if self.args.s3_pre_norm:
                            a_event = self.norm_a(a_event)
                            v_event = self.norm_v(v_event)

                        a_event_new = self.mhsa_a(a_event, a_event, a_event, a_mask) + self.mhsa_av(a_event, v_event, v_event, v_mask)
                        v_event_new = self.mhsa_v(v_event, v_event, v_event, v_mask) + self.mhsa_va(v_event, a_event, a_event, a_mask) 

                        a_event_new = F.dropout(a_event_new, self.dropout, training=self.training)
                        v_event_new = F.dropout(v_event_new, self.dropout, training=self.training)

                        a_event = a_event_new + a_shortcut
                        v_event = v_event_new + v_shortcut

                        if self.args.s3_post_norm:
                            a_event = self.norm_a(a_event)
                            v_event = self.norm_v(v_event)
                
                else:
                    raise NotImplementedError('more method will be update in future')

            else:
                a_mask = self.mhsa.get_mask(a_event_list, bs, num_cls)
                a_mask = a_mask.to('cuda')
                v_mask = self.mhsa.get_mask(v_event_list, bs, num_cls)
                v_mask = v_mask.to('cuda')
                
                a_shortcut = a_event
                v_shortcut = v_event
                
                if self.args.s3_pre_norm:
                    a_event = self.norm(a_event)
                    v_event = self.norm(v_event)
                
                a_event = self.mhsa(a_event, a_event, a_event, a_mask)
                v_event = self.mhsa(v_event, v_event, v_event, v_mask)
                
                

                a_event = F.dropout(a_event, self.dropout, training=self.training)
                v_event = F.dropout(v_event, self.dropout, training=self.training)

                a_event = a_event + a_shortcut
                v_event = v_event + v_shortcut

                if self.args.s3_post_norm:
                    a_event = self.norm(a_event)
                    v_event = self.norm(v_event)

        elif self.event_interaction_op =='none': 
            pass       
        
        else: 
            raise NotImplementedError('more models maybe update in future, now just have GAT and MHSA')
        
        
        self.a_event_new = a_event.detach()  
        self.v_event_new = v_event.detach()
        
        a_event_prob_list = []
        v_event_prob_list = []
        for i, j in zip(a_event_list[0], a_event_list[1]):
            a_event_prob = self.event_prob(fc_prob, a_event[i,j], j)
            a_event_prob_list.append(a_event_prob)
        
        for i, j in zip(v_event_list[0], v_event_list[1]):
            v_event_prob = self.event_prob(fc_prob, v_event[i,j], j)
            v_event_prob_list.append(v_event_prob)   
        
        
        a_prob, v_prob = self.re_cal_prob(a_event, v_event, a_event_list, v_event_list, a_prob, v_prob, frame_prob, x_a, x_v)
        
        return a_event_prob_list, v_event_prob_list, a_prob, v_prob 
    
    def event_prob(self, 
                fc_prob,  
                event_feature, 
                cls_index      
                ):
        if fc_prob == None:
            event_prob = torch.sigmoid(self.fc_prob(event_feature))[cls_index] 
        else:
            event_prob = torch.sigmoid(fc_prob(event_feature))[cls_index]
        return event_prob
    
    def event_adj(self, bs, num_cls, event_list):
        adj=np.zeros([bs,num_cls,num_cls])
        batch_list = [[] for _ in range(bs)]
        for i, j in zip(event_list[0], event_list[1]): 
            batch_list[i].append(j)
        
        for i,video_list in enumerate(batch_list):
            adj[i]=np.diag(np.ones(num_cls))
            row=np.zeros([num_cls,num_cls])
            col=np.zeros([num_cls,num_cls])
            row[video_list,:]=1 
            col[:,video_list]=1
            local_adj=np.where(row>col,col,row)  
            adj[i]=np.where(adj[i]>local_adj,adj[i],local_adj)
        
        return adj
    
    def cross_modal_event_adj(self, bs, num_cls, event_list1, event_list2):
        
        adj=np.zeros([bs,num_cls,num_cls])
        batch_list1 = [[] for _ in range(bs)]
        batch_list2 = [[] for _ in range(bs)]
        
        for i, j in zip(event_list1[0], event_list1[1]):  
            batch_list1[i].append(j)
        for i, j in zip(event_list2[0], event_list2[1]):  
            batch_list2[i].append(j)
        
        for i in range(bs):
            row=np.zeros([num_cls,num_cls])
            col=np.zeros([num_cls,num_cls])
            row[batch_list1[i],:]=1 

            col[:,batch_list2[i]]=1
            local_adj=np.where(row>col,col,row)  
            adj[i]=local_adj
        
        return adj

class ReCalProb(nn.Module):

    def __init__(self, model_dim=512, feature_detach=False, event_proj=False):
        super(ReCalProb, self).__init__()
        self.feature_detach = feature_detach
        self.event_proj = event_proj
        if event_proj:
            self.fc = nn.Linear(model_dim, model_dim)

    def forward(self, a_event, v_event, a_event_list, v_event_list, a_prob, v_prob, frame_prob, x_a, x_v):
        
        if self.event_proj:  
            a_event = self.fc(a_event)
            v_event = self.fc(v_event)

        a_prob_s3 = torch.clone(a_prob)  
        v_prob_s3 = torch.clone(v_prob)
        if self.feature_detach:
            a_event = a_event.detach()

        sim_a = cosine_distance(x_a, a_event)  
        att_a = torch.softmax(sim_a, dim=1)
        
        
        temporal_prob_a = att_a * frame_prob[:, :, 0, :]
        a_prob_new = temporal_prob_a.sum(dim=1)
        a_prob_s3[[a_event_list[0], a_event_list[1]]] = a_prob_new[[a_event_list[0], a_event_list[1]]] 

        if self.feature_detach:
            v_event = v_event.detach()
        sim_v = cosine_distance(x_v, v_event)
        att_v = torch.softmax(sim_v, dim=1)
        
        
        temporal_prob_v = att_v * frame_prob[:, :, 1, :]
        v_prob_new = temporal_prob_v.sum(dim=1)
        v_prob_s3[[v_event_list[0], v_event_list[1]]] = v_prob_new[[v_event_list[0], v_event_list[1]]] 
    
        return a_prob_s3, v_prob_s3
        
        
        
        






