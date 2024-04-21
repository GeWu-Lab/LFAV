import time

import torch

def train(args, model, train_loader, optimizer_list, criterion, epoch, writer):
    if args.num_stages == 1:
        stg1_optimizer = optimizer_list[0]
    elif args.num_stages == 2:
        stg1_optimizer, stg2_optimizer = optimizer_list
    elif args.num_stages == 3:
        stg1_optimizer, stg2_optimizer, stg3_optimizer = optimizer_list
    else:
        raise NotImplementedError

    model.train()

    print("\n------------- train -------------")



    with torch.autograd.set_detect_anomaly(True):

        for batch_idx, sample in enumerate(train_loader):
            start = time.time()
            s1_av_loss = 0 # s1 video loss scalar
            s2_av_loss = 0 # s2 video loss scalar
            s2_event_loss = 0 # s2 event loss scalar
            s3_event_loss = 0
            all_batch_idx = batch_idx + len(train_loader)*(epoch-1) # for tensorboard

            if args.label_format == 'video':
                audio, video, video_st, target = sample['audio'].to('cuda'), sample['video_s'].to('cuda'), sample[
                    'video_st'].to('cuda'), sample['label']
                target_video = target[0].type(torch.FloatTensor).to('cuda')
                if args.real_av_labels:
                    target_a = target[1].type(torch.FloatTensor).to('cuda')
                    target_v = target[2].type(torch.FloatTensor).to('cuda')
                else:
                    # label smoothing
                    a = 1.0
                    v = 0.9
                    target_a = a * target_video + (1 - a) * 0.5
                    target_v = v * target_video + (1 - v) * 0.5

            if args.num_stages >= 1:
                stg1_optimizer.zero_grad()
                if args.num_stages >= 2:
                    stg2_optimizer.zero_grad()
                    if args.num_stages == 3:
                        stg3_optimizer.zero_grad()

            if args.label_format == 'video':
                if args.num_stages == 1:
                    a_prob, v_prob, _ = model(audio, video, video_st)
                elif args.num_stages == 2:
                    a_prob, v_prob, _, ag_prob, vg_prob, _, a_event_prob_list, a_evnet_list, \
                    v_event_prob_list, v_evnet_list = model(audio, video, video_st) # if just stage2, do not need a_event and v_event
                elif args.num_stages == 3:
                    a_prob, v_prob, _, ag_prob, vg_prob, _, a_event_prob_list, a_evnet_list, \
                    v_event_prob_list, v_evnet_list, a_event_prob_s3, v_event_prob_s3, \
                    a_prob_s3, v_prob_s3 = model(audio, video, video_st)
                else:
                    raise NotImplementedError('just have 3 stages')
                forward_end = time.time()
                a_prob.clamp_(min=1e-7, max=1 - 1e-7)
                v_prob.clamp_(min=1e-7, max=1 - 1e-7)
                if args.num_stages >= 2:
                    ag_prob.clamp_(min=1e-7, max=1 - 1e-7)
                    vg_prob.clamp_(min=1e-7, max=1 - 1e-7)
                if args.num_stages >= 3:
                    a_prob_s3.clamp_(min=1e-7, max=1 - 1e-7)
                    v_prob_s3.clamp_(min=1e-7, max=1 - 1e-7)

                if args.num_stages >= 1:
                    stage1_loss = criterion(a_prob, target_a) + criterion(v_prob, target_v)
                    s1_av_loss = stage1_loss.item()
                    writer.add_scalar('stage1_loss', s1_av_loss, all_batch_idx)
                
                    if args.num_stages >= 2:
                        if args.just_event_loss:
                            stage2_loss=None
                            s2_av_loss = 0
                            writer.add_scalar('stage2_av_loss', s2_av_loss, all_batch_idx)

                        else:
                            stage2_loss=criterion(ag_prob, target_a) + criterion(vg_prob, target_v)
                            s2_av_loss = stage2_loss.item()
                            writer.add_scalar('stage2_av_loss', s2_av_loss, all_batch_idx)

                        if args.stg2_evnet_loss_weight > 0:
                            if args.el_warm_up_epoch > 0: # weight of event loss, soft warm-up
                                stg2_event_loss_weight = min(args.stg2_evnet_loss_weight, args.stg2_evnet_loss_weight * (epoch-1) / args.el_warm_up_epoch)
                            elif args.el_warm_up_epoch == 0:
                                stg2_event_loss_weight = args.stg2_evnet_loss_weight
                            else: # hard warm up
                                if epoch <= abs(args.el_warm_up_epoch):
                                    stg2_event_loss_weight = 0
                                else:
                                    stg2_event_loss_weight = args.stg2_evnet_loss_weight


                            if len(a_evnet_list[0])>0:
                                s2_event_loss_a = stg2_event_loss_weight*event_loss(criterion, a_event_prob_list, a_evnet_list, target_a)
                                s2_event_loss += s2_event_loss_a.item()
                                if stage2_loss==None:
                                    stage2_loss = s2_event_loss_a
                                else:
                                    stage2_loss += s2_event_loss_a
                            if len(v_evnet_list[0])>0:
                                s2_event_loss_v = stg2_event_loss_weight*event_loss(criterion, v_event_prob_list, v_evnet_list, target_v)
                                s2_event_loss += s2_event_loss_v.item()
                                if stage2_loss==None:
                                    stage2_loss = s2_event_loss_v
                                else:
                                    stage2_loss += s2_event_loss_v
                            writer.add_scalar('stage2_event_loss', s2_event_loss, all_batch_idx)
                        
                        # --------------------stage3 loss ------------------------------
                        if args.num_stages >= 3: # if have stage3, must use event loss, stage3 just have event loss
                            if args.s3_el_warm_up_epoch > 0: # weight of event loss, soft warm-up
                                s3_event_loss_weight = min(args.stg3_event_loss_weight, args.stg3_event_loss_weight * (epoch-1) / args.s3_el_warm_up_epoch)
                                s3_av_loss_weight = min(args.stg3_av_loss_weight, args.stg3_av_loss_weight * (epoch-1) / args.s3_el_warm_up_epoch)
                            elif args.s3_el_warm_up_epoch == 0:
                                s3_event_loss_weight = args.stg3_event_loss_weight
                                s3_av_loss_weight = args.stg3_av_loss_weight
                            else: # hard warm up
                                if epoch <= abs(args.s3_el_warm_up_epoch):
                                    s3_event_loss_weight = 0.0
                                    s3_av_loss_weight = 0.0
                                else:
                                    s3_event_loss_weight = args.stg3_event_loss_weight
                                    s3_av_loss_weight = args.stg3_av_loss_weight

                            stage3_loss = s3_av_loss_weight * (criterion(a_prob_s3, target_a) + criterion(v_prob_s3, target_v))
                            s3_av_loss = stage3_loss.item()
                            writer.add_scalar('stage3_av_loss', s3_av_loss, all_batch_idx)
                            if len(a_evnet_list[0])>0:  # event loss
                                s3_event_loss_a = s3_event_loss_weight * event_loss(criterion, a_event_prob_s3, a_evnet_list, target_a)
                                s3_event_loss += s3_event_loss_a.item()  # tensorboard use
                                stage3_loss += s3_event_loss_a

                            if len(v_evnet_list[0])>0:  # event loss
                                s3_event_loss_v = s3_event_loss_weight * event_loss(criterion, v_event_prob_s3, v_evnet_list, target_v)   
                                s3_event_loss += s3_event_loss_v.item()  
                                stage3_loss += s3_event_loss_v
                            
                            loss = stage1_loss + stage2_loss + stage3_loss

                            writer.add_scalar('stage3_event_loss', s3_event_loss, all_batch_idx)
                            # --------------------end stage3 loss ------------------------------
                        
                        else:
                            if stage2_loss==None:
                                loss = stage1_loss
                                stage2_loss=torch.tensor(0)
                            else:
                                loss = stage1_loss+stage2_loss
                    else:
                        loss = stage1_loss
            
            else:
                raise NotImplementedError

            loss.backward()
            if args.num_stages >= 1:
                stg1_optimizer.step()
                if args.num_stages >= 2:
                    stg2_optimizer.step()
                    if args.num_stages == 3:
                        stg3_optimizer.step()

            end = time.time()
            if batch_idx % args.log_interval == 0:
                if args.num_stages == 1:
                    print('Train Epoch: {} '
                          '[{}/{} ({:.0f}%)]\t'
                          'Stage1 Loss: {:.6f}\t'
                          'ForwardTime: {:.2f}\t'
                          'BackwardTime: {:.2f}'.format(epoch,
                                                        batch_idx * len(audio), len(train_loader.dataset),
                                                        100. * batch_idx / len(train_loader),
                                                        stage1_loss.item(),
                                                        forward_end - start,
                                                        end - forward_end))
                elif args.num_stages == 2:
                    print('Train Epoch: {} '
                          '[{}/{} ({:.0f}%)]\t'
                          'Stage1 Loss: {:.6f}\t'
                          'Stage2 Loss: {:.6f}\t'
                          'ForwardTime: {:.2f}\t'
                          'BackwardTime: {:.2f}'.format(epoch,
                                                        batch_idx * len(audio), len(train_loader.dataset),
                                                        100. * batch_idx / len(train_loader),
                                                        stage1_loss.item(),
                                                        stage2_loss.item(),
                                                        forward_end - start,
                                                        end - forward_end))
                elif args.num_stages == 3:
                    print('Train Epoch: {} '
                          '[{}/{} ({:.0f}%)]\t'
                          'Stage1 Loss: {:.6f}\t'
                          'Stage2 Loss: {:.6f}\t'
                          'Stage3 Loss: {:.6f}\t'
                          'ForwardTime: {:.2f}\t'
                          'BackwardTime: {:.2f}'.format(epoch,
                                                        batch_idx * len(audio), len(train_loader.dataset),
                                                        100. * batch_idx / len(train_loader),
                                                        stage1_loss.item(),
                                                        stage2_loss.item(),
                                                        stage3_loss.item(),
                                                        forward_end - start,
                                                        end - forward_end))
                else:
                    raise NotImplementedError

def event_loss(criterion, event_prob_list, event_list, target_m):
    # event_prob_list, event_list, m_prob
    event_prob=torch.stack(event_prob_list)
    event_prob.clamp_(min=1e-7, max=1 - 1e-7)

    if len(event_list) > 2:
        event_list = event_list[:2]
    target_list=target_m[event_list]
    return criterion(event_prob,target_list)