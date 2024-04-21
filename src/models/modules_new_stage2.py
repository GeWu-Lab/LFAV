from copy import deepcopy
from math import cos
from traceback import FrameSummary
import torch
import torch.nn as nn
import numpy as np



class MILPooling(nn.Module):
    def __init__(self, model_dim=512, num_cls=35):
        super(MILPooling, self).__init__()
        self.fc_prob = nn.Linear(model_dim, num_cls)
        self.fc_frame_att = nn.Linear(model_dim, num_cls)
    
    def forward(self, a, v):
        x = torch.cat([a.unsqueeze(-2), v.unsqueeze(-2)], dim=-2)  # (b, t, 2, dim)
        frame_prob = torch.sigmoid(self.fc_prob(x))

        frame_att = torch.softmax(self.fc_frame_att(x), dim=1)  # temporal attention w
        temporal_prob = (frame_att * frame_prob)
        a_prob = temporal_prob[:, :, 0, :].sum(dim=1)
        v_prob = temporal_prob[:, :, 1, :].sum(dim=1)

        return a_prob, v_prob, frame_prob, frame_att
    
    def event_prob(self,       
                event_feature, # new event feature
                cls_index      # index of class
                ):
        event_prob = torch.sigmoid(self.fc_prob(event_feature))[cls_index] # share weight (frame prob and event prob)

        return event_prob

class GraphFinetune():
    # add and delete node of subgraph

    def __init__(self, 
                pred_th = 0.5,   # threshold of predict prob
                cos_th = 0.9    # threshold of cos sim
                ):             
        self.pred_th = pred_th
        self.cos_th = cos_th
    
    def get_event_feature(self,
                        feature,       # feature, just one video, one modal
                        pool_method,   # method of get event feature
                        class_graph,   
                        frame_att=None,     # attention of each frame, just use when event_feature=='att'
                        keepdim=False        
                        ):
        if pool_method=='att': 
            assert frame_att is not None # frame_att used when pool_method = 'att'
            if frame_att.requires_grad:
                frame_att = frame_att.detach()
        
        if pool_method == 'avg':
            event_feature = torch.sum(feature[class_graph, :], dim=0, keepdim=keepdim)/len(class_graph)
        elif pool_method=='att':       
            event_feature = feature[class_graph, :] * frame_att[class_graph].unsqueeze(1)
            event_feature = torch.sum(event_feature, dim=0, keepdim=keepdim) / torch.sum(frame_att[class_graph])
        else:
            raise NotImplementedError

        return event_feature




