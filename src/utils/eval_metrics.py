import numpy as np


def Precision(X_pre, X_gt):
    N = len(X_pre)
    p = 0.0
    for i in range(N):
        x = X_pre[i, :]
        y = X_gt[i, :]
        p += np.sum(x * y) / np.sum(x)
    return p / N


def Recall(X_pre, X_gt):
    N = len(X_pre)
    p = 0.0
    for i in range(N):
        x = X_pre[i, :]
        y = X_gt[i, :]
        p += np.sum(x * y) / np.sum(y)
    return p / N


def F1(X_pre, X_gt):
    N = len(X_pre)
    p = 0
    for i in range(N):
        x = X_pre[i, :]
        y = X_gt[i, :]
        p += 2 * np.sum(x * y) / (np.sum(x) + np.sum(y))
    return p / N

def event_level(SO_a, SO_v, SO_av, GT_a, GT_v, GT_av, T, n_cls=35):
    
    N = n_cls
    event_p_a = [None for _ in range(n_cls)]
    event_gt_a = [None for _ in range(n_cls)]
    event_p_v = [None for _ in range(n_cls)]
    event_gt_v = [None for _ in range(n_cls)]
    event_p_av = [None for _ in range(n_cls)]
    event_gt_av = [None for _ in range(n_cls)]

    TP_a = np.zeros(n_cls)
    TP_v = np.zeros(n_cls)
    TP_av = np.zeros(n_cls)

    FP_a = np.zeros(n_cls)
    FP_v = np.zeros(n_cls)
    FP_av = np.zeros(n_cls)

    FN_a = np.zeros(n_cls)
    FN_v = np.zeros(n_cls)
    FN_av = np.zeros(n_cls)

    for n in range(N):
        seq_pred = SO_a[n, :]
        if np.sum(seq_pred) != 0:
            x = extract_event(seq_pred, n, T)
            event_p_a[n] = x
        seq_gt = GT_a[n, :]
        if np.sum(seq_gt) != 0:
            x = extract_event(seq_gt, n, T)
            event_gt_a[n] = x

        seq_pred = SO_v[n, :]
        if np.sum(seq_pred) != 0:
            x = extract_event(seq_pred, n, T)
            event_p_v[n] = x
        seq_gt = GT_v[n, :]
        if np.sum(seq_gt) != 0:
            x = extract_event(seq_gt, n, T)
            event_gt_v[n] = x

        seq_pred = SO_av[n, :]
        if np.sum(seq_pred) != 0:
            x = extract_event(seq_pred, n, T)
            event_p_av[n] = x

        seq_gt = GT_av[n, :]
        if np.sum(seq_gt) != 0:
            x = extract_event(seq_gt, n, T)
            event_gt_av[n] = x

        
        tp, fp, fn = event_wise_metric(event_p_a[n], event_gt_a[n])
        TP_a[n] += tp
        FP_a[n] += fp
        FN_a[n] += fn

        
        tp, fp, fn = event_wise_metric(event_p_v[n], event_gt_v[n])
        TP_v[n] += tp
        FP_v[n] += fp
        FN_v[n] += fn

        
        tp, fp, fn = event_wise_metric(event_p_av[n], event_gt_av[n])
        TP_av[n] += tp
        FP_av[n] += fp
        FN_av[n] += fn

    TP = TP_a + TP_v
    FN = FN_a + FN_v
    FP = FP_a + FP_v

    n = len(FP_a)
    F_a = []
    for ii in range(n):
        if (TP_a + FP_a)[ii] != 0 or (TP_a + FN_a)[ii] != 0:
            F_a.append(2 * TP_a[ii] / (2 * TP_a[ii] + (FN_a + FP_a)[ii]))

    F_v = []
    for ii in range(n):
        if (TP_v + FP_v)[ii] != 0 or (TP_v + FN_v)[ii] != 0:
            F_v.append(2 * TP_v[ii] / (2 * TP_v[ii] + (FN_v + FP_v)[ii]))

    F = []
    for ii in range(n):
        if (TP + FP)[ii] != 0 or (TP + FN)[ii] != 0:
            F.append(2 * TP[ii] / (2 * TP[ii] + (FN + FP)[ii]))

    F_av = []
    for ii in range(n):
        if (TP_av + FP_av)[ii] != 0 or (TP_av + FN_av)[ii] != 0:
            F_av.append(2 * TP_av[ii] / (2 * TP_av[ii] + (FN_av + FP_av)[ii]))

    if len(F_a) == 0:
        f_a = 1.0  
    else:
        f_a = (sum(F_a) / len(F_a))

    if len(F_v) == 0:
        f_v = 1.0  
    else:
        f_v = (sum(F_v) / len(F_v))

    if len(F) == 0:
        f = 1.0  
    else:
        f = (sum(F) / len(F))
    if len(F_av) == 0:
        f_av = 1.0  
    else:
        f_av = (sum(F_av) / len(F_av))

    return f_a, f_v, f, f_av

def segment_level(SO_a, SO_v, SO_av, GT_a, GT_v, GT_av):
    
    TP_a = np.sum(SO_a * GT_a, axis=1)
    FN_a = np.sum((1 - SO_a) * GT_a, axis=1)
    FP_a = np.sum(SO_a * (1 - GT_a), axis=1)

    n = len(FP_a)
    F_a = []
    for ii in range(n):
        if (TP_a + FP_a)[ii] != 0 or (TP_a + FN_a)[ii] != 0:
            F_a.append(2 * TP_a[ii] / (2 * TP_a[ii] + (FN_a + FP_a)[ii]))

    TP_v = np.sum(SO_v * GT_v, axis=1)
    FN_v = np.sum((1 - SO_v) * GT_v, axis=1)
    FP_v = np.sum(SO_v * (1 - GT_v), axis=1)
    F_v = []
    for ii in range(n):
        if (TP_v + FP_v)[ii] != 0 or (TP_v + FN_v)[ii] != 0:
            F_v.append(2 * TP_v[ii] / (2 * TP_v[ii] + (FN_v + FP_v)[ii]))

    TP = TP_a + TP_v
    FN = FN_a + FN_v
    FP = FP_a + FP_v

    n = len(FP)

    F = []
    for ii in range(n):
        if (TP + FP)[ii] != 0 or (TP + FN)[ii] != 0:
            F.append(2 * TP[ii] / (2 * TP[ii] + (FN + FP)[ii]))

    TP_av = np.sum(SO_av * GT_av, axis=1)
    FN_av = np.sum((1 - SO_av) * GT_av, axis=1)
    FP_av = np.sum(SO_av * (1 - GT_av), axis=1)
    n = len(FP_av)
    F_av = []
    for ii in range(n):
        if (TP_av + FP_av)[ii] != 0 or (TP_av + FN_av)[ii] != 0:
            F_av.append(2 * TP_av[ii] / (2 * TP_av[ii] + (FN_av + FP_av)[ii]))

    if len(F_a) == 0:
        f_a = 1.0  
    else:
        f_a = (sum(F_a) / len(F_a))

    if len(F_v) == 0:
        f_v = 1.0  
    else:
        f_v = (sum(F_v) / len(F_v))

    if len(F) == 0:
        f = 1.0  
    else:
        f = (sum(F) / len(F))
    if len(F_av) == 0:
        f_av = 1.0  
    else:
        f_av = (sum(F_av) / len(F_av))

    return f_a, f_v, f, f_av


def to_vec(start, end, t):
    x = np.zeros(t)
    for i in range(start, end):
        x[i] = 1
    return x


def extract_event(seq, n, T):
    x = []
    i = 0
    while i < T:
        if seq[i] == 1:
            start = i
            if i + 1 == T:
                i = i + 1
                end = i
                x.append(to_vec(start, end, T))
                break

            for j in range(i + 1, T):
                if seq[j] != 1:
                    i = j + 1  
                    end = j
                    x.append(to_vec(start, end, T))
                    break
                else:
                    i = j + 1
                    if i == T:
                        end = i
                        x.append(to_vec(start, end, T))
                        break
        else:
            i += 1
    return x


def event_wise_metric(event_p, event_gt):
    TP = 0
    FP = 0
    FN = 0

    if event_p is not None:
        num_event = len(event_p)
        for i in range(num_event):
            x1 = event_p[i]
            if event_gt is not None:
                nn = len(event_gt)
                flag = True
                for j in range(nn):
                    x2 = event_gt[j]
                    if np.sum(x1 * x2) >= 0.5 * np.sum(x1 + x2 - x1 * x2):  
                        TP += 1
                        flag = False
                        break
                if flag:
                    FP += 1
            else:
                FP += 1

    if event_gt is not None:
        num_event = len(event_gt)
        for i in range(num_event):
            x1 = event_gt[i]
            if event_p is not None:
                nn = len(event_p)
                flag = True
                for j in range(nn):
                    x2 = event_p[j]
                    if np.sum(x1 * x2) >= 0.5 * np.sum(x1 + x2 - x1 * x2):  
                        flag = False
                        break
                if flag:
                    FN += 1
            else:
                FN += 1
    return TP, FP, FN



def segment_map(prob_list,gt_list):
    
    

    num_cls=prob_list[0].shape[0]
    assert num_cls==35 

    map=[] 
    for i in range(num_cls):
        prec=[] 
        rec=[] 
        TP=0
        FP=0
        prob=np.concatenate([prob_list[ii][i] for ii in range(len(prob_list))])
        gt=np.concatenate([gt_list[ii][i] for ii in range(len(gt_list))])
        gt_num=np.sum(gt) 
        conf_index=np.argsort(prob) 
        conf_index=conf_index[::-1] 
        for index in conf_index:
            
            if gt[index]==1: 
                TP+=1
            else:
                FP+=1
            prec.append(TP/(TP+FP))
            rec.append(TP/gt_num)
    
        prec=np.array(prec)
        rec=np.array(rec)
        ap=interpolated_prec_rec(prec,rec)
        map.append(ap)
    return sum(map)/len(map)


def interpolated_prec_rec(prec, rec):
    """Interpolated AP - VOCdevkit from VOC 2011.
    """
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap

