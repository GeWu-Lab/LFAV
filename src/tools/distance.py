# calculate distance of features

import torch
import numpy as np

def cosine_distance(x, y, eps=1e-9):
    # cosine distance,
    # x and y are 3d tensors, (b, t, dim) or (b, num_cls, dim)
    x_norm=torch.norm(x,dim=-1,keepdim=True)
    y_norm=torch.norm(y,dim=-1,keepdim=True)
    xy_norm=x_norm*y_norm.transpose(-2,-1)
    xy_dot=torch.matmul(x,y.transpose(-2, -1))
    pairwise_distance=xy_dot/(xy_norm+eps)
    return pairwise_distance