"""
Network in stage two. It is based on graph attention network to achieve feature aggregation in event.
"""
from sys import modules
from copy import deepcopy
import torch

from .graph_modules import *
from .modules_new_stage2 import *
import time
from .get_multi_adj import get_adj, get_batch_adj
import numpy as np
from tools.distance import cosine_distance



class SnippetGAT(nn.Module):
    def __init__(self,
                 args,
                 model_dim=512,
                 snippet_graph_op='attn',
                 edge_threshold=0.7,
                 mask_update=False,
                 num_classes=35,
                 output_layer='fc',
                 graph_finetune=True
                 ):
        super(SnippetGAT, self).__init__()

        self.mask_update = mask_update
        self.num_classes = num_classes
        self.output_layer = output_layer
        self.graph_finetune = graph_finetune
        self.args=args  
        self.model_dim = model_dim

        if snippet_graph_op == 'conv':
            raise NotImplementedError('do not use now')
            
            
        elif snippet_graph_op == 'attn':  
            self.snippet_graph_op = DyGAT(input_dim=model_dim, residual=args.gat_residual)
            
        else:
            raise NotImplementedError("Incorrect graph operation {} ! "
                                      "Only between 'conv' and 'attn' !".format(snippet_graph_op))

        self.temporal_pooling = MILPooling(model_dim, num_classes)
        self.gfine=GraphFinetune(pred_th = args.add_sb_pred_th, cos_th = args.add_sb_cos_th)

    def forward(self, x_a, x_v, s1_frame_prob):
        
        
        s1_frame_prob_np = s1_frame_prob.detach().cpu().numpy() 
        self.s1_frame_prob = s1_frame_prob_np
        bs = x_a.shape[0]
        num_cls = s1_frame_prob_np.shape[-1] 

        a_event_prob_list = []  
        a_event_list = [[], [], []]  
        a_event=torch.zeros([bs, num_cls, self.model_dim], requires_grad=True).to('cuda')
        v_event_prob_list = []
        v_event_list = [[], [], []]        
        v_event=torch.zeros([bs, num_cls, self.model_dim], requires_grad=True).to('cuda')
        
        adj, bs_a_sb_list, bs_v_sb_list=get_batch_adj(s1_frame_prob_np, th=self.args.extract_sb_th, min_length=1, event_split=True, adj_mode=self.args.adj_mode) 
        
        self.a_class_graph_old = deepcopy(bs_a_sb_list)
        self.v_class_graph_old = deepcopy(bs_v_sb_list)

        adj=torch.tensor(adj).to('cuda')  

        x_a , x_v= self.snippet_graph_op(x_a, x_v, adj) 
        
        self.x_a=x_a.detach() 
        self.x_v=x_v.detach() 
                
        a_prob, v_prob, frame_prob, frame_att = self.temporal_pooling(x_a, x_v) 
        
        self.frame_att = frame_att.detach() 
        
        if self.graph_finetune:
            for i in range(bs):
                a_class_graph = bs_a_sb_list[i] 
                v_class_graph = bs_v_sb_list[i]
                for j in range(len(a_class_graph)): 
                    if len(a_class_graph[j]) > 0:  
                        # if self.args.add_node:
                        #     a_class_graph[j] = self.gfine.add_node(x_a[i], a_class_graph[j], s1_frame_prob_np[i, :, 0, j], frame_att[i, :, 0, j], self.args.pool_method)
                        event_feature = self.gfine.get_event_feature(x_a[i], self.args.pool_method, a_class_graph[j], frame_att = frame_att[i, :, 0, j])
                        event_prob = self.temporal_pooling.event_prob(event_feature, j)
                        
                        a_event_prob_list.append(event_prob) 
                        a_event_list[0].append(i) 
                        a_event_list[1].append(j) 
                        a_event_list[2].append(sum(a_class_graph[j])/len(a_class_graph[j])) 
                        a_event[i,j]=event_feature 
                    
                    if len(v_class_graph[j]) > 0:  
                        # if self.args.add_node:
                        #     v_class_graph[j] = self.gfine.add_node(x_v[i], v_class_graph[j], s1_frame_prob_np[i, :, 1, j], frame_att[i, :, 1, j], self.args.pool_method)
                        event_feature = self.gfine.get_event_feature(x_v[i], self.args.pool_method, v_class_graph[j], frame_att = frame_att[i, :, 1, j])
                        event_prob = self.temporal_pooling.event_prob(event_feature, j)
                        
                        v_event_prob_list.append(event_prob) 
                        v_event_list[0].append(i) 
                        v_event_list[1].append(j) 
                        v_event_list[2].append(sum(v_class_graph[j])/len(v_class_graph[j])) 
                        v_event[i,j]=event_feature 
        
        self.a_class_graph_new = bs_a_sb_list
        self.v_class_graph_new = bs_v_sb_list

        return a_prob, v_prob, frame_prob, x_a, x_v, a_event_prob_list, a_event_list, v_event_prob_list, v_event_list, a_event, v_event
