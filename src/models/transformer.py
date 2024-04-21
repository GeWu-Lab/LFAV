import torch
import torch.nn as nn

import math

from .base_models import BasicWindowTransformer, BasicPaddedWindowTransformer
from .modules import MultiInputSequential


class TransformerWithinModal(nn.Module):
    def __init__(self, dim=512, num_heads=4, qkv_bias=True, qk_scale=None,
                 attn_drop=0.2, proj_drop=0.2, temperature=1.):
        super(TransformerWithinModal, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5  
        self.temperature = temperature

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.pre_norm = nn.LayerNorm(dim)
        self.post_norm = nn.LayerNorm(dim)

        self.softmax = nn.Softmax(dim=-1)  

    def forward(self, x):
        shortcut = x
        x = self.pre_norm(x)

        b, t, dim = x.size()
        qkv = self.qkv(x).reshape(b, t, 3, self.num_heads, dim // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  

        attn = self.softmax(attn / self.temperature)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b, t, dim)  
        x = self.proj(x)
        x = self.proj_drop(x)

        x = x + shortcut
        x = self.post_norm(x)
        return x


class TransformerCrossModal(nn.Module):
    def __init__(self, dim=512, num_heads=4, qkv_bias=True, qk_scale=None,
                 attn_drop=0.2, proj_drop=0.2, temperature=1.):
        super(TransformerCrossModal, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5  
        self.temperature = temperature

        self.q = nn.Linear(dim, dim * 1, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.pre_norm = nn.LayerNorm(dim)
        self.post_norm = nn.LayerNorm(dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        shortcut = x
        x = self.pre_norm(x)

        b, t, dim = x.size()
        q = self.q(x).reshape(b, t, self.num_heads, dim // self.num_heads)
        q = q.permute(0, 2, 1, 3)  

        kv = self.kv(y).reshape(b, t, 2, self.num_heads, dim // self.num_heads)
        kv = kv.permute(2, 0, 3, 1, 4)  
        k, v = kv[0], kv[1]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  

        attn = self.softmax(attn / self.temperature)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b, t, dim)  
        x = self.proj(x)
        x = self.proj_drop(x)

        x = x + shortcut  
        x = self.post_norm(x)
        return x


class WindowTransformerWithinModal(BasicPaddedWindowTransformer):
    def __init__(self, dim=512, window_size=2,
                 num_heads=4, qkv_bias=True, qk_scale=None,
                 attn_drop=0.2, proj_drop=0.2, window_shift=True, temperature=1.):
        super(WindowTransformerWithinModal, self).__init__()

        self.dim = dim
        self.window_size = window_size
        self.window_shift = window_shift

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5  
        self.temperature = temperature

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.pre_norm = nn.LayerNorm(dim)
        self.post_norm = nn.LayerNorm(dim)

        self.softmax = nn.Softmax(dim=-1)  

    def forward(self, x):

        x = self.window_forward(x)

        if self.window_shift:
            x = self.sequence_shift(x)
            x = self.window_forward(x)
            x = self.sequence_inverse_shift(x)

        return x

    def window_forward(self, x):

        shortcut = x
        x = self.pre_norm(x)

        if self.window_size > 0:
            x, n_win = self.window_partition(x)  

        b_times_n_win, win_size, dim = x.size()
        qkv = self.qkv(x).reshape(b_times_n_win, win_size, 3, self.num_heads, dim // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  

        attn = self.softmax(attn / self.temperature)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_times_n_win, win_size, dim)  
        x = self.proj(x)
        x = self.proj_drop(x)

        if self.window_size > 0:
            x = self.window_reverse(x, n_win)  

        x = x + shortcut
        x = self.post_norm(x)

        return x


class WindowTransformerCrossModal(BasicPaddedWindowTransformer):
    def __init__(self, dim=512, window_size=2,
                 num_heads=4, qkv_bias=True, qk_scale=None,
                 attn_drop=0.2, proj_drop=0.2, window_shift=True, temperature=1.):
        super(WindowTransformerCrossModal, self).__init__()

        self.dim = dim
        self.window_size = window_size
        self.window_shift = window_shift

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5  
        self.temperature = temperature

        self.q = nn.Linear(dim, dim * 1, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.pre_norm = nn.LayerNorm(dim)
        self.post_norm = nn.LayerNorm(dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        x = self.window_forward(x, y)

        if self.window_shift:
            x = self.sequence_shift(x)
            y = self.sequence_shift(y)
            x = self.window_forward(x, y)
            x = self.sequence_inverse_shift(x)
        return x

    def window_forward(self, x, y):
        shortcut = x
        x = self.pre_norm(x)
        y = self.pre_norm(y)  

        if self.window_size > 0:
            x, n_win_x = self.window_partition(x)  
            y, n_win_y = self.window_partition(y)
            assert n_win_x == n_win_y, 'Length inconsistency between modalities !!!'

        b_times_n_win, win_size, dim = x.size()
        q = self.q(x).reshape(b_times_n_win, win_size, self.num_heads, dim // self.num_heads)
        q = q.permute(0, 2, 1, 3)  

        kv = self.kv(y).reshape(b_times_n_win, win_size, 2, self.num_heads, dim // self.num_heads)
        kv = kv.permute(2, 0, 3, 1, 4)  
        k, v = kv[0], kv[1]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  

        attn = self.softmax(attn / self.temperature)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_times_n_win, win_size, dim)  
        x = self.proj(x)
        x = self.proj_drop(x)

        if self.window_size > 0:
            x = self.window_reverse(x, n_win_x)  

        x = x + shortcut  
        x = self.post_norm(x)
        return x


class HybridWindowTransformerLayer(nn.Module):
    def __init__(self, args, model_dim=512, window_size=2, window_shift=True, num_heads=8, temperature=1.):
        super(HybridWindowTransformerLayer, self).__init__()

        use_window = window_size > 0
        self.args = args
        if use_window:
            self.within_modal_transformer_a = WindowTransformerWithinModal(dim=model_dim,
                                                                           window_size=window_size,
                                                                           window_shift=window_shift,
                                                                           num_heads=num_heads,
                                                                           temperature=temperature)
            self.within_modal_transformer_v = WindowTransformerWithinModal(dim=model_dim,
                                                                           window_size=window_size,
                                                                           window_shift=window_shift,
                                                                           num_heads=num_heads,
                                                                           temperature=temperature)
            self.cross_modal_transformer = WindowTransformerCrossModal(dim=model_dim,
                                                                       window_size=window_size,
                                                                       window_shift=window_shift,
                                                                       num_heads=num_heads,
                                                                       temperature=temperature)
            
            
            if self.args.s1_attn == 'self':
                self.within_modal_transformer_av = WindowTransformerWithinModal(dim=model_dim,
                                                                        window_size=window_size,
                                                                        window_shift=window_shift,
                                                                        num_heads=num_heads,
                                                                        temperature=temperature)
            
            if self.args.s1_attn == 'cm':
                self.cross_modal_transformer_a = WindowTransformerCrossModal(dim=model_dim,
                                                                       window_size=window_size,
                                                                       window_shift=window_shift,
                                                                       num_heads=num_heads,
                                                                       temperature=temperature)
                
                self.cross_modal_transformer_v = WindowTransformerCrossModal(dim=model_dim,
                                                                       window_size=window_size,
                                                                       window_shift=window_shift,
                                                                       num_heads=num_heads,
                                                                       temperature=temperature)

        else:
            self.within_modal_transformer_a = TransformerWithinModal(dim=model_dim,
                                                                     num_heads=num_heads,
                                                                     temperature=temperature)
            self.within_modal_transformer_v = TransformerWithinModal(dim=model_dim,
                                                                     num_heads=num_heads,
                                                                     temperature=temperature)
            self.cross_modal_transformer = TransformerCrossModal(dim=model_dim,
                                                                 num_heads=num_heads,
                                                                 temperature=temperature)

    def forward(self, x, y):
        if self.args.s1_attn == 'all':
            x = self.within_modal_transformer_a(x)
            y = self.within_modal_transformer_v(y)

            x_cross = self.cross_modal_transformer(x, y)
            y_cross = self.cross_modal_transformer(y, x)

            x = x + x_cross
            y = y + y_cross
        
        elif self.args.s1_attn == 'self':
            x = self.within_modal_transformer_a(x)
            y = self.within_modal_transformer_v(y)

            x_self = self.within_modal_transformer_av(x)
            y_self = self.within_modal_transformer_av(y)

            x = x + x_self
            y = y + y_self
        
        elif self.args.s1_attn == 'cm':

            x_c = self.cross_modal_transformer_a(x, y)
            y_c = self.cross_modal_transformer_v(y, x)

            x_cross = self.cross_modal_transformer(x_c, y_c)
            y_cross = self.cross_modal_transformer(y_c, x_c)

            x = x_c + x_cross
            y = y_c + y_cross
        
        elif self.args.s1_attn == 'none':
            pass

        return x, y


class HybridWindowTransformer(nn.Module):
    def __init__(self,
                 args,
                 model_dim=512,
                 num_heads=8,
                 temperature=1.,
                 num_hierarchy=3,
                 basic_window_size=2,
                 window_shift=True,
                 feature_flow='sequential'):
        super(HybridWindowTransformer, self).__init__()

        self.feature_flow = feature_flow
        self.num_hierarchy = num_hierarchy

        index = list(range(num_hierarchy))
        self.window_size_list = list(map(lambda x: basic_window_size ** (x + 1), index))
        
        self.args = args

        if feature_flow == 'sequential':
            transformer_layers = []
            for i in range(num_hierarchy):
                transformer_layers.append(HybridWindowTransformerLayer(args=args,
                                                                       model_dim=model_dim,
                                                                       window_size=self.window_size_list[i],
                                                                       window_shift=window_shift,
                                                                       num_heads=num_heads,
                                                                       temperature=temperature))
            self.multiscale_hybrid_transformer = MultiInputSequential(*transformer_layers)

        else:
            self.multiscale_hybrid_transformer = nn.ModuleList()
            for i in range(num_hierarchy):
                self.multiscale_hybrid_transformer.append(
                    HybridWindowTransformerLayer(args=args,
                                                 model_dim=model_dim,
                                                 window_size=self.window_size_list[i], 
                                                 window_shift=window_shift,
                                                 num_heads=num_heads,
                                                 temperature=temperature)
                )

            if feature_flow == 'ada_weight':
                self.gate = nn.Sequential(nn.Linear(model_dim, 128),
                                          nn.ReLU(),
                                          nn.Linear(128, model_dim))
                self.dim_reduction_x = nn.Linear(model_dim * num_hierarchy, model_dim)
                self.dim_reduction_y = nn.Linear(model_dim * num_hierarchy, model_dim)
                self.leaky_relu = nn.LeakyReLU()

            elif feature_flow == 'dense_connected':
                self.in_layer_dim_reduction_x = nn.Linear(model_dim * 2, model_dim)
                self.in_layer_dim_reduction_y = nn.Linear(model_dim * 2, model_dim)

                self.dim_reduction_x = nn.Linear(model_dim * (num_hierarchy + 1), model_dim)
                self.dim_reduction_y = nn.Linear(model_dim * (num_hierarchy + 1), model_dim)

                self.leaky_relu = nn.LeakyReLU()

            else:
                raise NotImplementedError('Incorrect feature flow !!! Got {} !!!'.format(feature_flow))

    def forward(self, x, y):

        if self.feature_flow == 'sequential':
            x, y = self.multiscale_hybrid_transformer(x, y)

        elif self.feature_flow == 'ada_weight':
            feature_list_x = []
            feature_list_y = []
            for i in range(self.num_hierarchy):
                x, y = self.multiscale_hybrid_transformer[i](x, y)
                feature_list_x.append(x)
                feature_list_y.append(y)
            x = torch.cat(feature_list_x, dim=-1)  
            y = torch.cat(feature_list_y, dim=-1)
            x = self.dim_reduction_x(x)
            x = self.leaky_relu(x)
            y = self.dim_reduction_y(y)
            y = self.leaky_relu(y)

        elif self.feature_flow == 'dense_connected':
            
            feature_list_x = [x]
            feature_list_y = [y]
            for i in range(self.num_hierarchy):
                new_x, new_y = self.multiscale_hybrid_transformer[i](x, y)

                x = torch.cat((x, new_x), dim=-1)
                y = torch.cat((y, new_y), dim=-1)

                
                x = self.in_layer_dim_reduction_x(x)
                x = self.leaky_relu(x)
                y = self.in_layer_dim_reduction_y(y)
                y = self.leaky_relu(y)

                feature_list_x.append(x)
                feature_list_y.append(y)

            x = torch.cat(feature_list_x, dim=-1)
            y = torch.cat(feature_list_y, dim=-1)

            x = self.dim_reduction_x(x)
            x = self.leaky_relu(x)
            y = self.dim_reduction_y(y)
            y = self.leaky_relu(y)

        else:
            raise NotImplementedError('Incorrect feature flow !!! Got {} !!!'.format(self.feature_flow))

        return x, y
