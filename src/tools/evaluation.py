from __future__ import print_function

import os
from termios import TIOCPKT_DOSTOP

from requests import TooManyRedirects

import numpy as np
import pandas as pd
import torch

from utils import CATEGORIES
from utils.eval_metrics import segment_level, event_level, segment_map


def evaluation(args, model, model_name, val_loader,
               eval_a_weak_file, eval_v_weak_file,
               eval_a_csv_file, eval_v_csv_file):
    """

    Args:
        args: arguments of evaluation
        model: model for evaluation
        model_name: model name
        val_loader: validation dataloader
        eval_a_weak_file: evaluation csv file
        eval_v_weak_file: evaluation csv file
        eval_a_csv_file: audio evaluation csv file
        eval_v_csv_file: visual evaluation csv file

    Returns:

    """
    
    categories = CATEGORIES

    n_categories = len(categories)
    assert n_categories == args.num_classes
    # print_subgraph=False
    model.eval()

    # load annotations
    df_a_w = pd.read_csv(eval_a_weak_file, header=0, sep='\t')
    df_v_w = pd.read_csv(eval_v_weak_file, header=0, sep='\t')
    df_a = pd.read_csv(eval_a_csv_file, header=0, sep='\t')
    df_v = pd.read_csv(eval_v_csv_file, header=0, sep='\t')

    id_to_idx = {id: index for index, id in enumerate(categories)}
    F_seg_a = []
    F_seg_v = []
    F_seg = []
    F_seg_av = []
    F_event_a = []
    F_event_v = []
    F_event = []
    F_event_av = []

    pa_list=[]
    pv_list=[]
    pav_list=[]
    gta_list=[]
    gtv_list=[]
    gtav_list=[]
        
    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            name, audio, video, video_st, target = sample['name'], sample['audio'].to('cuda'), sample['video_s'].to(
                'cuda'), sample['video_st'].to('cuda'), sample['label']

            if args.label_format == 'video':
                if args.num_stages == 1:
                    a_prob, v_prob, frame_prob = model(audio, video, video_st, name[0])
                elif args.num_stages == 2:
                    a_prob, v_prob, frame_prob, ag_prob, vg_prob, g_frame_prob, _, _, _, _ = model(audio, video, video_st) # # if just stage2, do not need a_event and v_event
                elif args.num_stages == 3:
                    a_prob, v_prob, frame_prob, ag_prob, vg_prob, g_frame_prob, _, _, _, _, _, _, _, _  = model(audio, video, video_st) # 这一行修改了输出使输出变量数匹配
                else:
                    pass
            if args.real_av_labels:
                assert args.label_format == 'video', 'real av labels only exist when label format is video !'
                if args.num_stages==1:
                    o_a = (a_prob.cpu().detach().numpy() >= 0.5).astype(np.int_)
                    o_v = (v_prob.cpu().detach().numpy() >= 0.5).astype(np.int_)

                    Pa = frame_prob[0, :, 0, :].cpu().detach().numpy()
                    Pv = frame_prob[0, :, 1, :].cpu().detach().numpy()

                    pa_list.append(np.transpose(Pa))
                    pv_list.append(np.transpose(Pv))
                    pav_list.append(np.transpose(Pa)*np.transpose(Pv))

                elif args.num_stages>=2: 
                    '''
                    stage3 just add an event loss, for evaluation, we need predict of each snippet and class,
                    do not need event prob, so s2 and s3 will share the same ouput in evaluation
                    '''
                    o_a = (ag_prob.cpu().detach().numpy() >= 0.5).astype(np.int_)
                    o_v = (vg_prob.cpu().detach().numpy() >= 0.5).astype(np.int_)

                    Pa = g_frame_prob[0, :, 0, :].cpu().detach().numpy()
                    Pv = g_frame_prob[0, :, 1, :].cpu().detach().numpy()

                    pa_list.append(np.transpose(Pa))
                    pv_list.append(np.transpose(Pv))
                    pav_list.append(np.transpose(Pa)*np.transpose(Pv))
                    
                    # ---------frame prob of s1 -----------
                    # model.stage2.s1_frame_prob  (bs, t, 2, num_cls) 
                    pas1=model.stage2.s1_frame_prob[0, :, 0, :] # (t, num_cls)
                    pas1=np.transpose(pas1)
                    
                    pvs1=model.stage2.s1_frame_prob[0, :, 1, :]
                    pvs1=np.transpose(pvs1)

                    # ---------frame att -----------
                    frame_att_a = model.stage2.frame_att[0, :, 0, :]
                    frame_att_v = model.stage2.frame_att[0, :, 1, :]
                    # print(frame_att_a.shape)
                    # ---------end-----------
                
                else:
                    raise NotImplementedError('num of stages must > 0')

                repeat_len = audio.size()[1]

                Pa = (Pa >= 0.5).astype(np.int_) * np.repeat(o_a, repeats=repeat_len, axis=0)
                Pv = (Pv >= 0.5).astype(np.int_) * np.repeat(o_v, repeats=repeat_len, axis=0)
            else:

                raise NotImplementedError('no need')

            # save result
            save_path = os.path.join(args.eval_output_path, model_name)
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            save_name = name[0] + ".txt"
            file_tmp = os.path.join(save_path, save_name)
            # eval_result = open(file_tmp, 'w')

            # ************************** extract audio GT labels **************************
            GT_a = np.zeros((n_categories, repeat_len))
            GT_v = np.zeros((n_categories, repeat_len))

            df_vid_a = df_a.loc[df_a['filename'] == df_a_w.loc[batch_idx, :][0]]
            filenames = df_vid_a["filename"]
            events = df_vid_a["event_labels"]
            onsets = df_vid_a["onset"]
            offsets = df_vid_a["offset"]
            num = len(filenames)
            if num > 0:
                for i in range(num):
                    x1 = int(onsets[df_vid_a.index[i]])
                    x2 = int(offsets[df_vid_a.index[i]])
                    event = events[df_vid_a.index[i]]
                    idx = id_to_idx[event]
                    GT_a[idx, x1:x2+1] = 1
                    

            # ************************** extract visual GT labels **************************
            df_vid_v = df_v.loc[df_v['filename'] == df_v_w.loc[batch_idx, :][0]]
            filenames = df_vid_v["filename"]
            events = df_vid_v["event_labels"]
            onsets = df_vid_v["onset"]
            offsets = df_vid_v["offset"]
            num = len(filenames)
            if num > 0:
                for i in range(num):
                    x1 = int(onsets[df_vid_v.index[i]])
                    x2 = int(offsets[df_vid_v.index[i]])
                    event = events[df_vid_v.index[i]]
                    idx = id_to_idx[event]
                    GT_v[idx, x1:x2+1] = 1

            # ************************** obtain audiovisual GT labels **************************
            GT_av = GT_a * GT_v


            gta_list.append(GT_a) # num_cls, t
            gtv_list.append(GT_v)
            gtav_list.append(GT_av)

            # obtain prediction matrices
            SO_a = np.transpose(Pa) 
            SO_v = np.transpose(Pv)
            SO_av = SO_a * SO_v

            # segment-level F1 scores
            f_a, f_v, f, f_av = segment_level(SO_a, SO_v, SO_av, GT_a, GT_v, GT_av)
            F_seg_a.append(f_a)
            F_seg_v.append(f_v)
            F_seg.append(f)
            F_seg_av.append(f_av)

            # event-level F1 scores
            f_a, f_v, f, f_av = event_level(SO_a, SO_v, SO_av, GT_a, GT_v, GT_av, repeat_len, 35)
            F_event_a.append(f_a)
            F_event_v.append(f_v)
            F_event.append(f)
            F_event_av.append(f_av)
            

    print("\n-------------------------Snippet-level MAP -------------------------")
    map_a=segment_map(pa_list,gta_list)
    print('snippet-level audio map: {:.2f}'.format(map_a*100))

    map_v=segment_map(pv_list,gtv_list)
    print('snippet-level visual map: {:.2f}'.format(map_v*100))

    map_av=segment_map(pav_list,gtav_list)
    print('snippet-level audio-visual map: {:.2f}'.format(map_av*100))

    map_avg = (map_a+map_v+map_av)/3
    print('snippet-level avg map: {:.2f}'.format(map_avg*100))


    print("\n------------------------- Event-level F1-------------------------")
    print('Audio Event Detection Event-level F1: {:.2f}'.format(100 * np.mean(np.array(F_event_a))))
    print('Visual Event Detection Event-level F1: {:.2f}'.format(100 * np.mean(np.array(F_event_v))))
    print('Audio-Visual Event Detection Event-level F1: {:.2f}'.format(100 * np.mean(np.array(F_event_av))))

    avg_type_event = (100 * np.mean(np.array(F_event_av)) + 100 * np.mean(np.array(F_event_a)) + 100 * np.mean(
        np.array(F_event_v))) / 3.
    print('Event-level Type@Avg. F1: {:.2f}'.format(avg_type_event))

    return (map_avg*100 + avg_type_event)/2
