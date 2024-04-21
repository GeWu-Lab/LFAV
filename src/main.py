from __future__ import print_function

import argparse
import os
import warnings
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from dataloader import MESSDataset, MESSDatasetNew, ToTensor, ToEqualLengthSample
from tools import train, evaluation
# from tools.evaluation_stat import evaluation_stat
from models import MultiStageNet
from utils import setup_seed
from tensorboardX import SummaryWriter

warnings.filterwarnings('ignore')
np.set_printoptions(threshold=np.inf)

print("\n------------------ MESS experiment -------------------------\n")


def main():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of '
                                                 'Multi-Event Scene Understanding with Sight and Sound')

    # feature paths
    parser.add_argument("--audio_feature_path", type=str, required=True,
                        help="audio feature path")
    parser.add_argument("--visual_feature_path", type=str, required=True,
                        help="2D visual feature path dir")
    parser.add_argument("--spatio_temporal_visual_feature_path", type=str, required=True,
                        help="spatio-temporal visual feature path")

    # label utility
    parser.add_argument("--label_format", type=str, default='video', choices=['video'],
                        help="use audio-visual separate weakly annotated labels or video level ones")
    parser.add_argument("--real_av_labels", action='store_true',
                        help="use real audio-visual separate weakly labels or label smoothing")
    # train_label paths
    parser.add_argument("--weak_label_train_video_level", type=str, default="label_path/train/train_weakly.txt")
    parser.add_argument("--weak_label_train_audio", type=str, default="label_path/train/train_audio_weakly.txt",
                        help="audio weak train csv file")
    parser.add_argument("--weak_label_train_visual", type=str, default="label_path/train/train_visual_weakly.txt",
                        help="visual weak train csv file")
    # val_label_paths
    parser.add_argument("--weak_label_val_audio", type=str, default="label_path/val/val_audio_weakly.csv",
                        help="audio weak test csv file")
    parser.add_argument("--weak_label_val_visual", type=str, default="label_path/val/val_visual_weakly.csv",
                        help="visual weak test csv file")
    parser.add_argument("--weak_label_val", type=str, default="label_path/val/val_weak_av.csv",
                        help="weak test csv file")
    parser.add_argument("--label_val_audio", type=str, default="label_path/val/val_audio.csv",
                        help="temporally fine-grained annotated validation csv file for audio")
    parser.add_argument("--label_val_visual", type=str, default="label_path/val/val_visual.csv",
                        help="temporally fine-grained annotated validation csv file for visual")

    # test_label_paths
    parser.add_argument("--weak_label_test_audio", type=str, default="label_path/test/test_audio_weakly.csv",
                        help="audio weak test csv file")
    parser.add_argument("--weak_label_test_visual", type=str, default="label_path/test/test_visual_weakly.csv",
                        help="visual weak test csv file")
    parser.add_argument("--weak_label_test", type=str, default="label_path/test/test_weak_av.csv",
                        help="weak test csv file")
    parser.add_argument("--label_test_audio", type=str, default="label_path/test/test_audio.csv",
                        help="temporally fine-grained annotated validation csv file for audio")
    parser.add_argument("--label_test_visual", type=str, default="label_path/test/test_visual.csv",
                        help="temporally fine-grained annotated validation csv file for visual")

    # training settings
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train (default: 30)')
    parser.add_argument('--stg1_lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--stg2_lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--stg3_lr', type=float, default=1e-4, help='learning rate (default: 3.3e-5)')
    parser.add_argument('--step_size', type=int, default=10, help='step size')
    parser.add_argument('--stg2_evnet_loss_weight', type=float, default=1.0, help='learning rate (default: 1e-4)')
    parser.add_argument('--stg3_event_loss_weight', type=float, default=1.0, help='learning rate (default: 1e-4)')
    parser.add_argument('--stg3_av_loss_weight', type=float, default=1.0, help='stg3 av loss weight')
    parser.add_argument("--just_event_loss", action='store_true',help="just use event loss in stage2")
    parser.add_argument("--el_warm_up_epoch", type=int, default=0, help="will be use in future")
    parser.add_argument("--s3_el_warm_up_epoch", type=int, default=0, help="will be use in future")
    parser.add_argument('--num_classes', type=int, default=35)


    # model settings
    parser.add_argument("--num_stages", type=int, default=1, help="number of stages in the model")
    # 1. Segment Level Network
    parser.add_argument("--transformer_dim", type=int, default=512,
                        help="dimension in multiscale transformer")
    parser.add_argument("--transformer_num_heads", type=int, default=8,
                        help="number of heads in multiscale transformer")
    parser.add_argument("--mask_generator_type", type=str, default='conv', choices=['conv', 'attn'],
                        help="network type of the mask generator")
    parser.add_argument("--transformer_temperature", type=float, default=1.,
                        help="temperature in softmax of multiscale transformer")
    parser.add_argument("--num_transformer_layers", type=int, default=6,
                        help="number of layers in multiscale transformer")
    parser.add_argument("--window_shift", action='store_true',
                        help="whether to shift window within a layer")
    parser.add_argument("--basic_window_size", type=int, default=2,
                        help="the size of smallest window in multiscale transformer")
    parser.add_argument("--flow_between_layers", type=str, default='sequential',
                        choices=['sequential', 'ada_weight', 'dense_connected'],
                        help="control the feature flow in multiscale window transformer")
    parser.add_argument("--s1_attn", type=str, default='all', choices=['all', 'self', 'cm', 'none'], 
                        help="the size of smallest window in multiscale transformer")

    # 2. GAT Network
    # hyp-parameters in GAT is fixed, not in args
    parser.add_argument("--extract_sb_th", type=float, default=0.7,
                        help="threshold when extract subgraph accroding to the stage1")
    parser.add_argument("--add_sb_cos_th", type=float, default=0.9,
                        help="cos sim th when add node")
    parser.add_argument("--add_sb_pred_th", type=float, default=0.5,
                        help="cos sim th when add node")
    parser.add_argument("--pool_method", type=str, default='avg',choices=['avg', 'att'],
                        help="method of get event feature")
    parser.add_argument("--gat_residual", action='store_true',help="use skip connection in GAT")
    parser.add_argument("--cross_modal", action='store_true',help="will be use in future")
    parser.add_argument("--adj_mode", default='local', choices=['local', 'global'], help="will be use in future")
    
    # prior GAT args, do not use new, will delete in future
    parser.add_argument("--gat_edge_threshold", type=float, default=0.5,
                        help="mask edge threshold before GAT")
    parser.add_argument("--graph_op_type", type=str, default='conv', choices=['conv', 'attn'],
                        help="network type of the graph operation")
    
    # 3 event interaction network
    parser.add_argument("--event_interaction_op", type=str, default='attn',
                        choices=['attn','mhsa','none'], help="operations of event interaction, now just can choose GAT, maybe add more in future")
    parser.add_argument("--s3_within_modal", action='store_true', help="within modal interaction in s3 event interaction network or not")
    parser.add_argument("--s3_cross_modal", action='store_true', help="cross modal in s3 event interaction network or not")
    parser.add_argument("--s3_residual", action='store_true',help="whether use skip connection in s3 event interaction net")
    parser.add_argument("--s3_share_fc", action='store_true',help="for get event prob in s3, use fc_prob in s2")
    parser.add_argument("--s3_gat_nheads", type=int, default=1, help='num heads of GAT in stage3')
    parser.add_argument("--s3_mhsa_nheads", type=int, default=1, help='num heads of MHSA in stage3')
    parser.add_argument("--s3_mhsa_pe", action='store_true' , help='use position encoding in S3 or not')
    parser.add_argument("--s3_gat_depth", type=int, default=2, help='depth of GAT in stage3, if 1, just have one output layer')
    parser.add_argument("--s3_dropout", type=float, default=0.0, help='dropout of S3')
    parser.add_argument("--s3_cm_method", type=str, default='concat', choices=['concat', 'add', 'sequential'], help='cross modal attention method of MHSA in s3')
    parser.add_argument("--s3_pre_norm", action='store_true' , help='LN before MHSA in s3')
    parser.add_argument("--s3_post_norm", action='store_true' , help='LN after MHSA and residual in s3')
    parser.add_argument("--s3_no_share_weight", action='store_true' , help="use a fc layer for event projection")
    parser.add_argument("--s3_share_cm", action='store_true' , help="self attention and cross-modal attention share weights")
    parser.add_argument("--s3_just_cm", action='store_true' , help="just use cross modal attention")
    parser.add_argument("--s3_feature_detach", action='store_true' , help='detach grad when reweight snippet attention for event feature')
    parser.add_argument("--s3_event_proj", action='store_true' , help="use a fc layer for event projection")
    parser.add_argument("--s3_attn", type=str, default='all', choices=['all', 'self', 'cm', 'none'], 
                        help="the size of smallest window in multiscale transformer")

    # other setting
    parser.add_argument("--train", action='store_true', help="train or test")
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=50,
                        help='how many batches to wait before logging training status')
    parser.add_argument("--model_path", type=str, default='ckpts/', help="path to save trained models")
    parser.add_argument("--resume", action='store_true', help="whether resume from a saved checkpoint")
    parser.add_argument("--eval_output_path", type=str, default='eval_output/', help="path to save evaluation")
    parser.add_argument('--gpu', type=str, default='1', help='gpu device number')
    parser.add_argument("--experiment_date", type=str, required=True, help="e.g., Apr15")

    parser.add_argument("--tensorsummary_name", type=str, default='', help="name of tensorboard summary file")
    parser.add_argument("--save_prob_dir", type=str, default='', help="save dir of predict prob")

    args = parser.parse_args()
    print(args)
    print('Experiment data: ', args.experiment_date)

    model_name = 'Multimodal-Multi-Event-Video-Analyzer_' \
                 'label_utility-{}_' \
                 'number_of_stages-{}_' \
                 'transformer_layers-{}_' \
                 'basic_window_size-{}_' \
                 'window_shift-{}_' \
                 'experiment_date-{}'.format(args.label_format,
                                             args.num_stages,
                                             args.num_transformer_layers,
                                             args.basic_window_size,
                                             args.window_shift,
                                             args.experiment_date)

    print('Model Name: \n', model_name)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    setup_seed(args.seed)
    np.set_printoptions(threshold=np.inf)

    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)
    if not os.path.exists(args.eval_output_path):
        os.mkdir(args.eval_output_path)

    device = torch.device('cuda:0')

    model = MultiStageNet(args,
                          num_stages=args.num_stages,
                          label_utility=args.label_format,
                          num_hierarchy=args.num_transformer_layers,
                          window_shift=args.window_shift,
                          basic_window_size=args.basic_window_size,
                          snippet_graph_op=args.graph_op_type,
                          gat_edge_threshold=args.gat_edge_threshold)

    model = model.to(device)

    if args.resume:
        if os.path.exists(args.model_path + model_name + ".pth"):
            print('resume from {}'.format(args.model_path + model_name + ".pth"))
            model.load_state_dict(torch.load(args.model_path + model_name + ".pth"))
        else:
            print('No resume checkpoint found! Train from scratch!')

    if args.train:

        if args.label_format == 'video':
            train_dataset = MESSDatasetNew(label=args.weak_label_train_video_level,
                                           label_a=args.weak_label_train_audio,
                                           label_v=args.weak_label_train_visual,
                                           audio_dir=args.audio_feature_path,
                                           video_dir=args.visual_feature_path,
                                           st_dir=args.spatio_temporal_visual_feature_path,
                                           transform=transforms.Compose([ToTensor(), ToEqualLengthSample()]))
        else:
            raise NotImplementedError

        val_dataset = MESSDataset(label=args.weak_label_val,
                                  audio_dir=args.audio_feature_path,
                                  video_dir=args.visual_feature_path,
                                  st_dir=args.spatio_temporal_visual_feature_path,
                                  transform=transforms.Compose([ToTensor()]))

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                  pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True,
                                drop_last=False)
        optimizer_list = []
        if args.num_stages >= 1:
            stg1_optimizer = optim.Adam(model.stage1.parameters(), lr=args.stg1_lr)
            stg1_scheduler = optim.lr_scheduler.StepLR(stg1_optimizer, step_size=args.step_size, gamma=0.1)
            optimizer_list.append(stg1_optimizer)
            if args.num_stages >= 2:
                stg2_optimizer = optim.Adam(model.stage2.parameters(), lr=args.stg2_lr)
                stg2_scheduler = optim.lr_scheduler.StepLR(stg2_optimizer, step_size=args.step_size, gamma=0.1)
                optimizer_list.append(stg2_optimizer)
                if args.num_stages >= 3:
                    stg3_optimizer = optim.Adam(model.stage3.parameters(), lr=args.stg3_lr)
                    stg3_scheduler = optim.lr_scheduler.StepLR(stg3_optimizer, step_size=args.step_size, gamma=0.1)
                    optimizer_list.append(stg3_optimizer)
        criterion = nn.BCELoss()

        best_F = 0
        if len(args.tensorsummary_name)==0:
            current_time = time.strftime("%Y-%m-%dT%H:%M", time.localtime()) # if not input name, use current time as name
        else:
            current_time = args.tensorsummary_name
        my_writer = SummaryWriter(log_dir='tensorboard_summary/'+current_time)
        # writer = SummaryWriter('./tensorboard_Result')
        
        for epoch in range(1, args.epochs + 1):

            train(args, model, train_loader, optimizer_list, criterion, epoch=epoch, writer=my_writer)

            if args.num_stages >= 1:
                stg1_scheduler.step(epoch)
                if args.num_stages >= 2:
                    stg2_scheduler.step(epoch)
                    if args.num_stages >= 3:
                        stg3_scheduler.step(epoch)
            
            print('\n')
            print('---------------start val--------------------')
            F = evaluation(args, model, model_name, val_loader, # val
                           args.weak_label_val_audio,
                           args.weak_label_val_visual,
                           args.label_val_audio,
                           args.label_val_visual)
            if F > best_F: # select best model in val dataset
                best_F = F
                torch.save(model.state_dict(), args.model_path + model_name + ".pth")
                print('save model, epoch: ', epoch)
            print('---------------end val--------------------')

    else: # test
        model.load_state_dict(torch.load(args.model_path + model_name + ".pth"), strict=False)

        test_dataset = MESSDataset(label=args.weak_label_test,
                                  audio_dir=args.audio_feature_path,
                                  video_dir=args.visual_feature_path,
                                  st_dir=args.spatio_temporal_visual_feature_path,
                                  transform=transforms.Compose([ToTensor()]))

        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True,
                                drop_last=False)

        print('\n')
        print('---------------start test--------------------')
        evaluation(args, model, model_name, test_loader, # test
                           args.weak_label_test_audio,
                           args.weak_label_test_visual,
                           args.label_test_audio,
                           args.label_test_visual)
        print('---------------end test--------------------')
        print('\n')


if __name__ == '__main__':
    main()
