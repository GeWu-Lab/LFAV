import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from utils import constant


def ids_to_multinomial(ids):
    """ label encoding

    Returns:
      1d array, multimonial representation, e.g. [1,0,1,0,0,...]
    """
    categories = constant.CATEGORIES

    id_to_idx = {id: index for index, id in enumerate(categories)}

    y = np.zeros(len(categories))

    for id in ids:
        try:
            index = id_to_idx[id.strip()]
            y[index] = 1
        except KeyError:
            if id == 'silent': 
                pass
            else:
                print('id: ', id)
                print('ids:', ids)
    return y


def ids_to_multinomial_null(ids):
    """ label encoding

    Returns:
      1d array, multimonial representation, e.g. [1,0,1,0,0,...]
    """
    categories = constant.CATEGORIES_NULL

    id_to_idx = {id: index for index, id in enumerate(categories)}

    y = np.zeros(len(categories))

    for id in ids:
        index = id_to_idx[id.strip()]
        y[index] = 1
    return y


class MESSDataset(Dataset):

    def __init__(self, label, audio_dir, video_dir, st_dir, transform=None):
        self.df = pd.read_csv(label, header=0, sep='\t')
        self.filenames = self.df["filename"]
        self.audio_dir = audio_dir
        self.video_dir = video_dir
        self.st_dir = st_dir
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        row = self.df.loc[idx, :]
        name = row[0][:11]
        audio = np.load(os.path.join(self.audio_dir, name + '.npy'))
        video_s = np.load(os.path.join(self.video_dir, name + '.npy'))
        video_st = np.load(os.path.join(self.st_dir, name + '.npy'))
        ids = row[-1].split(',')
        label = ids_to_multinomial(ids)

        
        sample = {'name': name, 'audio': audio, 'video_s': video_s, 'video_st': video_st, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

class MESSDatasetNew(Dataset):

    def __init__(self, label, label_a, label_v, audio_dir, video_dir, st_dir, transform=None):
        self.df = pd.read_csv(label, header=0, sep='\t')
        self.df_a = pd.read_csv(label_a, header=0, sep='\t')
        self.df_v = pd.read_csv(label_v, header=0, sep='\t')
        self.filenames = self.df_a["filename"]
        self.audio_dir = audio_dir
        self.video_dir = video_dir
        self.st_dir = st_dir
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        row = self.df.loc[idx, :]
        row_a = self.df_a.loc[idx, :]
        row_v = self.df_v.loc[idx, :]
        name = row_a[0][:11]

        audio = np.load(os.path.join(self.audio_dir, name + '.npy'))
        video_s = np.load(os.path.join(self.video_dir, name + '.npy'))
        video_st = np.load(os.path.join(self.st_dir, name + '.npy'))

        
        ids = row[-1].split(',')
        label = ids_to_multinomial(ids)
        
        
        ids_a = row_a[-1].split(',')
        label_a = ids_to_multinomial(ids_a)

        
        ids_v = row_v[-1].split(',')
        label_v = ids_to_multinomial(ids_v)

        label = [label, label_a, label_v]

        
        sample = {'name': name, 'audio': audio, 'video_s': video_s, 'video_st': video_st, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        
        return sample


class ToTensor(object):

    def __call__(self, sample):
        if len(sample) == 2:
            audio = sample['audio']
            label = sample['label']
            return {'audio': torch.from_numpy(audio), 'label': torch.from_numpy(label)}
        else:
            name = sample['name']
            audio = sample['audio']
            video_s = sample['video_s']
            video_st = sample['video_st']
            label = sample['label']

            return {'name': name, 'audio': torch.from_numpy(audio), 'video_s': torch.from_numpy(video_s),
                    'video_st': torch.from_numpy(video_st),
                    'label': label}


class ToEqualLength(object):

    def __init__(self, length=1000):

        self.length = length

    def __call__(self, sample):

        if len(sample) == 2:
            audio = sample['audio']
            label = sample['label']

            audio = audio.unsqueeze(0).permute(0, 2, 1).contiguous()
            audio = F.interpolate(audio, size=self.length, mode='linear')
            audio = audio.permute(0, 2, 1).contiguous().squeeze()

            return {'audio': audio, 'label': label}
        else:
            name = sample['name']
            audio = sample['audio']
            video_s = sample['video_s']
            video_st = sample['video_st']
            label = sample['label']

            audio = audio.unsqueeze(0).permute(0, 2, 1).contiguous()
            audio = F.interpolate(audio, size=self.length, mode='linear')
            audio = audio.permute(0, 2, 1).contiguous().squeeze()

            video_s = video_s.unsqueeze(0).permute(0, 2, 1).contiguous()
            video_s = F.interpolate(video_s, size=self.length, mode='linear')
            video_s = video_s.permute(0, 2, 1).contiguous().squeeze()

            video_st = video_st.unsqueeze(0).permute(0, 2, 1).contiguous()
            video_st = F.interpolate(video_st, size=self.length, mode='linear')
            video_st = video_st.permute(0, 2, 1).contiguous().squeeze()

            return {'name': name, 'audio': audio, 'video_s': video_s,
                    'video_st': video_st, 'label': label}


class ToEqualLengthSample(object):

    def __init__(self, length=200):

        self.length = length

    def __call__(self, sample):

        if len(sample) == 2:
            audio = sample['audio']
            label = sample['label']

            audio = audio.unsqueeze(0).permute(0, 2, 1).contiguous()
            audio = F.interpolate(audio, size=self.length, mode='linear')
            audio = audio.permute(0, 2, 1).contiguous().squeeze()

            return {'audio': audio, 'label': label}
        else:
            name = sample['name']
            audio = sample['audio']
            video_s = sample['video_s']
            video_st = sample['video_st']
            label = sample['label']

            seq_len = audio.size(0)
            if seq_len >= self.length:
                audio = self.downsample(audio, self.length)
                video_s = self.downsample(video_s, self.length)
                video_st = self.downsample(video_st, self.length)
            else:
                audio = self.upsample(audio, self.length)
                video_s = self.upsample(video_s, self.length)
                video_st = self.upsample(video_st, self.length)

            return {'name': name, 'audio': audio, 'video_s': video_s,
                    'video_st': video_st, 'label': label}

    @staticmethod
    def downsample(x, length):
        stride = x.size(0) // length
        sampled_x = x[::stride, :]
        sampled_x = sampled_x[:length, :]
        return sampled_x

    @staticmethod
    def upsample(x, length):
        x = x.unsqueeze(0).permute(0, 2, 1).contiguous()
        sampled_x = F.interpolate(x, length, mode='linear')
        sampled_x = sampled_x.permute(0, 2, 1).contiguous().squeeze()
        return sampled_x
