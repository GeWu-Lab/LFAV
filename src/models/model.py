import torch
import torch.nn as nn

from .stage_one import StackedWindowTransformer
from .stage_two_new import SnippetGAT
from .stage_three import EventInteractionNet
import time


class MultiStageNet(nn.Module):
    def __init__(self,
                 args,  
                 num_stages=3,
                 label_utility='video',
                 model_dim=512,
                 num_heads_in_transformer=4,
                 mask_generator_type='conv',
                 temperature_in_transformer=1,
                 num_hierarchy=6,
                 flow_across_layers='sequential',
                 window_shift=True,
                 basic_window_size=2,
                 num_classes=35,
                 snippet_graph_op='conv',
                 event_graph_op='attn',
                 gat_edge_threshold=0.5,
                 ):
        super(MultiStageNet, self).__init__()

        self.label_utility = label_utility

        self.num_stage = num_stages
        self.args = args

        self.stage1 = StackedWindowTransformer(args,
                                               label_utility=label_utility,
                                               model_dim=model_dim,
                                               num_heads_in_transformer=num_heads_in_transformer,
                                               mask_generator_type=mask_generator_type,
                                               temperature_in_transformer=temperature_in_transformer,
                                               num_hierarchy=num_hierarchy,
                                               flow_across_layers=flow_across_layers,
                                               window_shift=window_shift,
                                               basic_window_size=basic_window_size,
                                               num_classes=num_classes)

        if num_stages >= 2:
            self.stage2 = SnippetGAT(args,
                                     model_dim=model_dim,
                                     snippet_graph_op=snippet_graph_op,
                                     edge_threshold=gat_edge_threshold)

        if num_stages >= 3:
            self.stage3 = EventInteractionNet(args,
                                              model_dim=model_dim,
                                              event_interaction_op=args.event_interaction_op,
                                              num_classes=num_classes)

    def forward(self, audio, visual, visual_st, id=None):

        if self.label_utility == 'video':
            if self.num_stage >= 1: 
                a_prob, v_prob, frame_prob, x_a, x_v = self.stage1(audio, visual, visual_st)
                if self.num_stage >= 2:
                    g_a_prob, g_v_prob, g_frame_prob, x_a, x_v, a_event_prob_list,\
                    a_event_list, v_event_prob_list, v_event_list , a_event, v_event = self.stage2(x_a, x_v, frame_prob)
                    if self.num_stage >= 3:
                        if self.args.s3_share_fc: 
                            fc_prob = self.stage2.temporal_pooling.fc_prob
                        else:
                            fc_prob = None
                        a_event_prob_s3, v_event_prob_s3, a_prob_s3, v_prob_s3 = self.stage3(a_event, v_event, a_event_list, v_event_list, g_a_prob, g_v_prob, g_frame_prob, x_a, x_v, fc_prob) 
                        return a_prob, v_prob, frame_prob, \
                        g_a_prob, g_v_prob, g_frame_prob, a_event_prob_list, a_event_list, v_event_prob_list, v_event_list, \
                        a_event_prob_s3, v_event_prob_s3, a_prob_s3, v_prob_s3
                        
                    else:
                        return a_prob, v_prob, frame_prob, \
                        g_a_prob, g_v_prob, g_frame_prob, a_event_prob_list, a_event_list, v_event_prob_list, v_event_list 
                else:
                    return a_prob, v_prob, frame_prob
        else:
            raise NotImplementedError('Label format should only be video, '
                                      'but got {} !!! '.format(self.label_utility))
