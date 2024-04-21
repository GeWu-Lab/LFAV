"""
Network in stage one. It is based on stacked shifted window transformers to capture events with various durations.
"""
import os.path

import torch

from .modules import *
from .transformer import *


class StackedWindowTransformer(nn.Module):
    def __init__(self,
                 args,
                 label_utility='video',
                 model_dim=512,
                 num_heads_in_transformer=4,
                 mask_generator_type='conv',
                 temperature_in_transformer=1,
                 num_hierarchy=6,
                 flow_across_layers='sequential',
                 window_shift=True,
                 basic_window_size=2,
                 num_classes=35):
        super(StackedWindowTransformer, self).__init__()

        self.label_utility = label_utility
        self.num_classes = num_classes
        self.args = args

        
        self.fc_a = nn.Linear(128, model_dim)
        self.fc_v = nn.Linear(512, model_dim)
        self.fc_st = nn.Linear(512, model_dim)
        self.fc_fusion = nn.Linear(model_dim * 2, model_dim)
        
        
        self.multiscale_hybrid_transformer = HybridWindowTransformer(args,
                                                                     model_dim=model_dim,
                                                                     num_heads=num_heads_in_transformer,
                                                                     num_hierarchy=num_hierarchy,
                                                                     temperature=temperature_in_transformer,
                                                                     basic_window_size=basic_window_size,
                                                                     window_shift=window_shift,
                                                                     feature_flow=flow_across_layers)

        if label_utility == 'video':
            self.temporal_pooling = MILPooling(model_dim, num_classes)
        else:
            raise NotImplementedError('Label format should only be video, '
                                      'but got {} !!! '.format(self.label_utility))
        

    def forward(self, audio, visual, visual_st):
        
        x_a = self.fc_a(audio)
        vid_s = self.fc_v(visual)
        vid_st = self.fc_st(visual_st)
        x_v = torch.cat((vid_s, vid_st), dim=-1)
        x_v = self.fc_fusion(x_v)

        x_a, x_v = self.multiscale_hybrid_transformer(x_a, x_v)
        
        if self.label_utility == 'video':
            a_prob, v_prob, frame_prob = self.temporal_pooling(x_a, x_v)
            
            return a_prob, v_prob, frame_prob, x_a, x_v
        else:
            raise NotImplementedError('Label format should only be video, '
                                      'but got {} !!! '.format(self.label_utility))
