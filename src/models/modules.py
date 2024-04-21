from copy import deepcopy
import torch
import torch.nn as nn
import numpy as np


class MLP(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.2):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class MultiInputSequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class MILPooling(nn.Module): 
    def __init__(self, model_dim=512, num_cls=35):
        super(MILPooling, self).__init__()
        self.fc_prob = nn.Linear(model_dim, num_cls)
        self.fc_frame_att = nn.Linear(model_dim, num_cls)

    def forward(self, a, v):
        x = torch.cat([a.unsqueeze(-2), v.unsqueeze(-2)], dim=-2)  
        frame_prob = torch.sigmoid(self.fc_prob(x))

        frame_att = torch.softmax(self.fc_frame_att(x), dim=1)  
        temporal_prob = (frame_att * frame_prob)
        a_prob = temporal_prob[:, :, 0, :].sum(dim=1)
        v_prob = temporal_prob[:, :, 1, :].sum(dim=1)

        return a_prob, v_prob, frame_prob

