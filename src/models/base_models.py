

import torch
import torch.nn as nn


class BasicWindowTransformer(nn.Module):
    def __init__(self, dim=512, window_size=2):
        super(BasicWindowTransformer, self).__init__()

        self.dim = dim
        self.window_size = window_size

    def forward(self, *input):
        raise NotImplementedError

    def window_partition(self, original_input):
        """
        Args:
            original_input: (b, t, dim)
        Returns: (b*num_windows, window_size, dim)
        """
        b, t, dim = original_input.size()
        window_input = original_input.view(b, t // self.window_size, self.window_size, dim)
        num_windows = t // self.window_size
        window_input = window_input.view(-1, self.window_size, dim).contiguous()
        return window_input, num_windows

    def window_reverse(self, window_output, num_windows):
        """
        Args:
            window_output: (b*num_windows, window_size, dim)
            num_windows: int
        Returns: (b, t, dim)
        """
        b_times_n_win, window_size, dim = window_output.size()
        assert window_size == self.window_size, 'Inconsistency in window size !!!'
        window_output = window_output.view(-1, num_windows, window_size, dim)
        reverse_input = window_output.view(-1, num_windows * window_size, dim).contiguous()
        return reverse_input


class BasicPaddedWindowTransformer(nn.Module):
    def __init__(self, dim=512, window_size=2, window_shift=True, shift_stride=None):
        super(BasicPaddedWindowTransformer, self).__init__()

        self.dim = dim
        self.window_size = window_size
        self.window_shift = window_shift
        self.shift_stride = shift_stride or window_size // 2

        self.pad_flag = 0
        self.padded_len = 0

    def forward(self, *input):
        raise NotImplementedError

    def window_forward(self, *input):
        raise NotImplementedError

    def window_partition(self, original_input):
        """
        Args:
            original_input: (b, t, dim)
        Returns: (b*num_windows, window_size, dim)
        """
        self.pad_flag = 0
        b, t, dim = original_input.size()
        if t % self.window_size == 0:
            window_input = original_input.view(b, t // self.window_size, self.window_size, dim)
            num_windows = t // self.window_size
            window_input = window_input.view(-1, self.window_size, dim).contiguous()
        else:
            residual_seg_len = t % self.window_size
            padded_len = self.window_size - residual_seg_len
            self.padded_len = padded_len
            self.pad_flag = 1
            
            padding = original_input[:, -padded_len:]
            padded_input = torch.cat((original_input, padding), dim=1)

            num_windows = t // self.window_size + 1
            assert num_windows == (t + padded_len) // self.window_size, \
                'wrong padding !!!, got num_windows = {}, t = {}, ' \
                'padded len = {}, self.window_size = {}'.format(
                    num_windows, t, padded_len, self.window_size)
            window_input = padded_input.view(b, (t + padded_len) // self.window_size, self.window_size, dim)
            window_input = window_input.view(-1, self.window_size, dim).contiguous()

        return window_input, num_windows

    def window_reverse(self, window_output, num_windows):
        """
        Args:
            window_output: (b*num_windows, window_size, dim)
            num_windows: int
        Returns: (b, t, dim)
        """
        b_times_n_win, window_size, dim = window_output.size()
        assert window_size == self.window_size, 'Inconsistency in window size !!!'
        window_output = window_output.view(-1, num_windows, window_size, dim)
        reverse_input = window_output.view(-1, num_windows * window_size, dim).contiguous()

        if self.pad_flag != 0:
            reverse_input = reverse_input[:, :-self.padded_len]

        return reverse_input

    def sequence_shift(self, x):
        """
        Shift the input sequence to fit the window shift operation.
        E.g.,:
        1111 2222 3333 4444 --> 1122 2233 3344 4411
        """

        shifted_x = torch.roll(x, shifts=-self.shift_stride, dims=1)
        return shifted_x

    def sequence_inverse_shift(self, shifted_x):
        """
        Recover the original sequence w.r.t. the temporal order.
        E.g.,:
        1122 2233 3344 4411 --> 1111 2222 3333 4444
        """
        
        x = torch.roll(shifted_x, shifts=self.shift_stride, dims=1)
        return x
