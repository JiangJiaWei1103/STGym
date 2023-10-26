"""
Baseline method, SCINet [NeurIPS, 2022].
Author: ChunWei Shen

Reference:
* Paper: https://arxiv.org/abs/2106.09305
* Code: https://github.com/cure-lab/SCINet
"""
from typing import List, Dict, Any, Tuple, Union, Optional

import math
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from metadata import MTSFBerks
from torch.nn.modules.loss import _Loss
from utils.scaler import MaxScaler, StandardScaler

class SCINet(nn.Module):
    def __init__(
        self,
        net_params: Dict[str, Any],
        loss_params: Dict[str, Any]
    ):
        """
        SCINet.

        Parameters:
            dataset_name: dataset name
            t_window: lookback time window
            n_series: number of nodes
            num_stacks: number of SCINet block
            num_levels: SCINet block levels
            num_decoder_layer: number of decoder layers
            output_len: output sequence length
            concat_len: sequence length to concatenate with the output of the intermediate SCINet
            hid_size: hidden size of conv1d
            kernel_size: kernel size of conv1d
            groups: number of conv1d groups
            dropout: dropout ratio
            INN: whether to use interactive learning
            single_step_output_One: only output the single final step
            positionalEcoding: whether to add positional ecoding
            RIN: whether to add RIN
            single_step: only output the single final step (for loss)
            lastWeight: loss weight for final step
            criterion: criterion
        """
        super(SCINet, self).__init__()

        # Network parameters
        self.net_params = net_params
        self.loss_params = loss_params

        # hyperparameters of SCINet
        self.dataset_name = self.net_params['dataset_name']
        self.t_window = self.net_params['t_window']
        self.n_series = self.net_params['n_series']
        self.num_stacks  = self.net_params['num_stacks']
        self.num_levels = self.net_params['num_levels']
        self.num_decoder_layer = self.net_params['num_decoder_layer']
        self.output_len = self.net_params['output_len']
        self.concat_len = self.net_params['concat_len']
        self.hidden_size = self.net_params['hid_size']
        self.kernel_size = self.net_params['kernel_size']
        self.groups = self.net_params['groups']
        self.dropout = self.net_params['dropout']
        self.INN = self.net_params['INN']
        self.single_step_output_One = self.net_params['single_step_output_One']
        self.pe = self.net_params['positionalEcoding']
        self.RIN = self.net_params['RIN']
        # hyperparameters of loss function
        self.single_step = self.loss_params['single_step']
        self.lastWeight = self.loss_params['lastWeight']
        criterion = self.loss_params['criterion']

        if criterion == 'smooth_l1_loss':
            self.criterion = self.smooth_l1_loss
        else:
            self.criterion = None
        
        # SCINet
        self.blocks1 = _EncoderTree(
            in_planes=self.n_series,
            hidden_size=self.hidden_size,
            kernel_size=self.kernel_size,
            groups=self.groups,
            dropout=self.dropout,
            INN=self.INN,
            num_levels=self.num_levels)

        # Stacked SCINet if num_stacks > 1
        if self.num_stacks == 2:
            self.blocks2 = _EncoderTree(
                in_planes=self.n_series,
                num_levels=self.num_levels,
                kernel_size=self.kernel_size,
                dropout=self.dropout,
                groups=self.groups,
                hidden_size=self.hidden_size,
                INN=self.INN)

        self.stacks = self.num_stacks

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

        # output projection
        self.projection1 = nn.Conv1d(self.t_window, self.output_len, kernel_size=1, stride=1, bias=False)
        self.div_projection = nn.ModuleList()
        self.overlap_len = self.t_window // 4
        self.div_len = self.t_window // 6

        # output projection if number of decoder layer > 1
        if self.num_decoder_layer > 1:
            self.projection1 = nn.Linear(self.t_window, self.output_len)
            for layer_idx in range(self.num_decoder_layer-1):
                div_projection = nn.ModuleList()
                for i in range(6):
                    lens = min(i * self.div_len + self.overlap_len, self.t_window) - i * self.div_len
                    div_projection.append(nn.Linear(lens, self.div_len))
                self.div_projection.append(div_projection)

        # only output the N_th timestep
        if self.single_step_output_One:
            if self.stacks == 2:
                if self.concat_len:
                    self.projection2 = nn.Conv1d(
                        in_channels=self.concat_len + self.output_len,
                        out_channels=1,
                        kernel_size=1,
                        bias=False)
                else:
                    self.projection2 = nn.Conv1d(
                        in_channels=self.t_window + self.output_len,
                        out_channels=1,
                        kernel_size=1,
                        bias=False)
        # output all N timesteps
        else:
            if self.stacks == 2:
                if self.concat_len:
                    self.projection2 = nn.Conv1d(
                        in_channels=self.concat_len + self.output_len,
                        out_channels=self.output_len,
                        kernel_size=1,
                        bias=False)
                else:
                    self.projection2 = nn.Conv1d(
                        in_channels=self.t_window + self.output_len,
                        out_channels=self.output_len,
                        kernel_size=1,
                        bias=False)

        # Positional encoding
        self.pe_hidden_size = self.n_series
        if self.pe_hidden_size % 2 == 1:
            self.pe_hidden_size += 1
    
        num_timescales = self.pe_hidden_size // 2
        max_timescale = 10000.0
        min_timescale = 1.0

        log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / max(num_timescales - 1, 1))
        temp = torch.arange(num_timescales, dtype=torch.float32)
        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales, dtype=torch.float32) *
            -log_timescale_increment)
        self.register_buffer('inv_timescales', inv_timescales)

        # RIN Parameters
        if self.RIN:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, self.n_series))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, self.n_series))
    
    def get_position_encoding(
        self,
        x: Tensor
    ) -> Tensor:
        """
        position encoding.

        Parameters:
            x: input features

        Shape:
            x: (B, T, N)
            signal: (1, T, pe_hidden_size)
        """
        max_length = x.size()[1]
        position = torch.arange(max_length, dtype = torch.float32, device = x.device)
        temp1 = position.unsqueeze(1)               # (max_length, 1)
        temp2 = self.inv_timescales.unsqueeze(0)    # (1, num_timescales)
        scaled_time = temp1 * temp2                 # (max_length, num_timescales)
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim = 1)  # (T, N) or (T, N+1)
        signal = F.pad(signal, (0, 0, 0, self.pe_hidden_size % 2))
        signal = signal.view(1, max_length, self.pe_hidden_size)
    
        return signal

    def forward(
        self,
        x: Tensor,
        As: Optional[List[Tensor]] = None,
        **kwargs: Any,
    ) -> Tuple[Tensor, Union[Tensor, None], None]:
        """
        Forward pass.

        Parameters:
            x: input features

        Shape:
            x: (B, P, N, C)
        """
        x = x[..., 0]
        # evenly divided the input length into two parts. (e.g., 32 -> 16 -> 8 -> 4 for 3 levels)
        assert self.t_window % (np.power(2, self.num_levels)) == 0

        # Positional encoding
        if self.pe:
            pe = self.get_position_encoding(x)
            if pe.shape[2] > x.shape[2]:
                x += pe[:, :, :-1]
            else:
                x += self.get_position_encoding(x)

        # Activated when RIN is True
        if self.RIN:
            means = x.mean(1, keepdim=True).detach()    # mean
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)    # var
            x /= stdev
            x = x * self.affine_weight + self.affine_bias   # affine

        # the first stack
        res1 = x
        x = self.blocks1(x)
        x += res1
        # decoder layer
        if self.num_decoder_layer == 1:
            x = self.projection1(x)
        else:
            x = x.permute(0,2,1)
            for div_projection in self.div_projection:
                output = torch.zeros(x.shape, dtype = x.dtype).to(x.device)
                for i, div_layer in enumerate(div_projection):
                    div_x = x[:, :, i * self.div_len : min(i * self.div_len + self.overlap_len, self.t_window)]
                    output[:, :, i * self.div_len : (i + 1) * self.div_len] = div_layer(div_x)
                x = output
            x = self.projection1(x)
            x = x.permute(0,2,1)

        if self.stacks == 1:
            # reverse RIN
            if self.RIN:
                x = x - self.affine_bias
                x = x / (self.affine_weight + 1e-10)
                x = x * stdev
                x = x + means

            return x, None, None
        
        elif self.stacks == 2:
            MidOutPut = x
            if self.concat_len:
                x = torch.cat((res1[:, -self.concat_len:,:], x), dim=1)
            else:
                x = torch.cat((res1, x), dim=1)
            # the second stack
            res2 = x
            x = self.blocks2(x)
            x += res2
            x = self.projection2(x)
            
            # Reverse RIN
            if self.RIN:
                # Reverse MidOutPut
                MidOutPut = MidOutPut - self.affine_bias
                MidOutPut = MidOutPut / (self.affine_weight + 1e-10)
                MidOutPut = MidOutPut * stdev
                MidOutPut = MidOutPut + means
                # Reverse x
                x = x - self.affine_bias
                x = x / (self.affine_weight + 1e-10)
                x = x * stdev
                x = x + means

            return x, MidOutPut, None
        
    def smooth_l1_loss(
        self,
        y_pred: Tensor,
        y_true: Tensor,
        beta: float = 1. / 9,
        size_average: bool = True
    ) -> float:
        """
        Very similar to the smooth_l1_loss from pytorch,
        but with the extra beta parameter.
        """
        n = torch.abs(y_pred - y_true)
        cond = n < beta
        loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)

        if size_average:
            return loss.mean()
        
        return loss.sum()
        
    def train_loss(
        self,
        y_pred: Tensor,
        y_true: Tensor,
        Midoutput: Tensor,
        output2: Union[Tensor, None],
        scaler: Union[MaxScaler, StandardScaler],
        criterion: _Loss,
    ) -> float:
        """
        loss function.

        Parameters:
            y_pred: model prediction
            y_true: ground truth
            output1: Mid Output or None
            output2: None
            scaler: scaler
        """
        if self.criterion is None:
            self.criterion = criterion
            
        weight = torch.tensor(self.lastWeight).to(y_pred.device)

        if Midoutput is not None:
            Midoutput = scaler.inverse_transform(Midoutput)

        if self.single_step: # single step
            y_last = y_true[:, -1, :]
            loss_f = self.criterion(y_pred[:, -1, :], y_last)
            if self.stacks == 2:
                loss_m = self.criterion(Midoutput, y_true) / Midoutput.shape[1] # average results
        else:
            if self.lastWeight == 1.0:
                loss_f = self.criterion(y_pred, y_true)
                if self.stacks == 2:
                    loss_m = self.criterion(Midoutput, y_true)
            else:
                loss_f = self.criterion(y_pred[:, :-1, :], y_true[:, :-1, :] ) \
                        + weight * self.criterion(y_pred[:, -1:, :], y_true[:, -1:, :] )
                if self.stacks == 2:
                    loss_m = self.criterion(Midoutput[:, :-1, :] , y_true[:, :-1, :] ) \
                            + weight * self.criterion(Midoutput[:, -1:, :], y_true[:, -1:, :])
                    
        loss = loss_f
        if self.stacks == 2:
            loss += loss_m
        
        return loss
    
    def eval_loss(
        self,
        y_pred: Tensor,
        y_true: Tensor,
        Midoutput: Tensor,
        output2: Union[Tensor, None],
        scaler: Union[MaxScaler, StandardScaler],
        criterion: _Loss,
    ) -> Tuple[float, Tensor, Tensor]:
        """
        loss function.

        Parameters:
            y_pred: model prediction
            y_true: ground truth
            output1: Mid Output
            output2: None
            scaler: scaler
            criterion: criterion
        """
        y_pred_inv = scaler.inverse_transform(y_pred)
        y_true_inv = scaler.inverse_transform(y_true)
        if Midoutput is not None:
            Midoutput = scaler.inverse_transform(Midoutput)

        if self.dataset_name in MTSFBerks:
            y_pred = y_pred[:, -1, :]
            y_true = y_true[:, -1, :]
            y_pred_inv = y_pred_inv[:, -1, :]
            y_true_inv = y_true_inv[:, -1, :]
            loss = criterion(y_pred_inv, y_true_inv)
        else:
            loss = criterion(y_pred_inv, y_true_inv)

        return loss, y_pred, y_true

class _Splitting(nn.Module):
    """
    Downsamples the original sequence into two sub-sequences
    by separating the even and the odd elements.
    """
    def __init__(self):
        super(_Splitting, self).__init__()

    def forward(
        self,
        x: Tensor
    ) -> Tensor:
        """
        Forward pass.

        Parameters:
            x: input features

        Shape:
            x: (B, T, N)
        """
        even = x[:, ::2, :]
        odd = x[:, 1::2, :]

        return even, odd
    
class _SCIBlock(nn.Module):
    def __init__(
        self,
        in_planes: int,
        hidden_size: int = 1,
        kernel: int = 5,
        groups: int = 1,
        dropout: float = 0.5,
        splitting: bool = True,
        INN: bool = True
    ):
        """
        SCI-Block.

        Parameters:
            in_planes: input channels
            hidden_size: hidden size for Conv1d
            kernel: kernel size for Conv1d
            groups: groups for Conv1d
            dropout: dropout ratio
            splitting: whether to split
            INN: whether to use interactive learning
        """
        super(_SCIBlock, self).__init__()

        self.hidden_size = hidden_size
        self.kernel_size = kernel
        self.groups = groups
        self.dropout = dropout
        self.splitting = splitting
        self.INN = INN
        self.dilation = 1

        # size of the padding
        if self.kernel_size % 2 == 0:
            pad_l = self.dilation * (self.kernel_size - 2) // 2 + 1
            pad_r = self.dilation * (self.kernel_size) // 2 + 1
        else:
            pad_l = self.dilation * (self.kernel_size - 1) // 2 + 1
            pad_r = self.dilation * (self.kernel_size - 1) // 2 + 1

        self.split = _Splitting()

        modules_phi = []
        modules_psi = []
        modules_P = []
        modules_U = []
        prev_size = 1

        # convolutional module phi
        modules_phi += [
            nn.ReplicationPad1d(padding=(pad_l, pad_r)),
            nn.Conv1d(in_channels=in_planes * prev_size,
                      out_channels=int(in_planes * hidden_size),
                      kernel_size=self.kernel_size,
                      dilation=self.dilation,
                      stride=1,
                      groups=self.groups),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv1d(in_channels=int(in_planes * hidden_size),
                      out_channels=in_planes,
                      kernel_size=3,
                      stride=1,
                      groups=self.groups),
            nn.Tanh()
        ]

        # convolutional module psi
        modules_psi += [
            nn.ReplicationPad1d(padding=(pad_l, pad_r)),
            nn.Conv1d(in_channels=in_planes * prev_size, 
                      out_channels=int(in_planes * hidden_size),
                      kernel_size=self.kernel_size,
                      dilation=self.dilation,
                      stride=1,
                      groups=self.groups),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv1d(in_channels=int(in_planes * hidden_size),
                      out_channels=in_planes,
                      kernel_size=3,
                      stride=1,
                      groups=self.groups),
            nn.Tanh()
        ]

        # convolutional module P
        modules_P += [
            nn.ReplicationPad1d(padding=(pad_l, pad_r)),
            nn.Conv1d(in_channels=in_planes * prev_size,
                      out_channels=int(in_planes * hidden_size),
                      kernel_size=self.kernel_size,
                      dilation=self.dilation,
                      stride=1,
                      groups=self.groups),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv1d(in_channels=int(in_planes * hidden_size),
                      out_channels=in_planes,
                      kernel_size=3, 
                      stride=1, 
                      groups=self.groups),
            nn.Tanh()
        ]

        # convolutional module U
        modules_U += [
            nn.ReplicationPad1d(padding=(pad_l, pad_r)),
            nn.Conv1d(in_channels=in_planes * prev_size,
                      out_channels=int(in_planes * hidden_size),
                      kernel_size=self.kernel_size,
                      dilation=self.dilation,
                      stride=1,
                      groups=self.groups),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv1d(in_channels=int(in_planes * hidden_size),
                      out_channels=in_planes,
                      kernel_size=3,
                      stride=1,
                      groups=self.groups),
            nn.Tanh()
        ]
        
        self.phi = nn.Sequential(*modules_phi)
        self.psi = nn.Sequential(*modules_psi)
        self.P = nn.Sequential(*modules_P)
        self.U = nn.Sequential(*modules_U)

    def forward(
        self,
        x: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass.

        Parameters:
            x: input features
        
        Shape:
            x: (B, T, N)
            F_even_update: (B, N, T)
            F_odd_update: (B, N, T)
        """
        if self.splitting:
            # splitting
            (F_even, F_odd) = self.split(x)
        else:
            (F_even, F_odd) = x

        if self.INN:
            # interactive learning
            F_even = F_even.permute(0, 2, 1)    # (B, N, T)
            F_odd = F_odd.permute(0, 2, 1)      # (B, N, T)

            Fs_odd = F_odd.mul(torch.exp(self.phi(F_even)))
            Fs_even = F_even.mul(torch.exp(self.psi(F_odd)))

            F_even_update = Fs_even + self.U(Fs_odd)
            F_odd_update = Fs_odd - self.P(Fs_even)

            return (F_even_update, F_odd_update)
        else:
            F_even = F_even.permute(0, 2, 1)    # (B, N, T)
            F_odd = F_odd.permute(0, 2, 1)      # (B, N, T)

            F_odd_update = F_odd - self.P(F_even)
            F_even_update = F_even + self.U(Fs_odd)

            return (F_even_update, F_odd_update)
        
class _SCIBlockLevel(nn.Module):
    def __init__(
        self,
        in_planes: int,
        hidden_size: int,
        kernel_size: int,
        groups: int,
        dropout: float,
        INN: bool
    ):
        """
        SCI-Block Level.

        Parameters:
            in_planes: input channels
            hidden_size: hidden size for Conv1d
            kernel_size: kernel size for Conv1d
            groups: groups for Conv1d
            dropout: dropout ratio
            INN: whether to use interactive learning
        """
        super(_SCIBlockLevel, self).__init__()

        self.level = _SCIBlock(
            in_planes=in_planes,
            hidden_size=hidden_size,
            kernel=kernel_size,
            groups=groups,
            dropout=dropout,
            splitting=True,
            INN=INN)

    def forward(
        self,
        x: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass.

        Parameters:
            x: input features

        Shape:
            x: (B, T, N)
            F_even_update: (B, D, T)
            F_odd_update: (B, D, T)
        """

        F_even_update, F_odd_update = self.level(x)

        return F_even_update, F_odd_update

class _LevelSCINet(nn.Module):
    def __init__(
        self,
        in_planes: int,
        hidden_size: int,
        kernel_size: int,
        groups: int,
        dropout: float,
        INN: bool
    ):
        """
        SCINet Level.

        Parameters:
            in_planes: input channels
            hidden_size: hidden size for Conv1d
            kernel_size: kernel size for Conv1d
            groups: groups for Conv1d
            dropout: dropout ratio
            INN: whether to use interactive learning
        """
        super(_LevelSCINet, self).__init__()

        self.interact = _SCIBlockLevel(
            in_planes=in_planes,
            hidden_size=hidden_size,
            kernel_size=kernel_size,
            groups=groups,
            dropout=dropout,
            INN=INN)

    def forward(
        self,
        x: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass.

        Parameters:
            x: input features

        Shape:
            x: (B, T, N)
            F_even_update: (B, T, D)
            F_odd_update: (B, T, D)
        """
        F_even_update, F_odd_update = self.interact(x)
        F_even_update = F_even_update.permute(0, 2, 1)
        F_odd_update = F_odd_update.permute(0, 2, 1)

        return F_even_update, F_odd_update
    
class _SCINet_Tree(nn.Module):
    def __init__(
        self,
        in_planes: int,
        hidden_size: int,
        kernel_size: int,
        groups: int,
        dropout: float,
        INN: bool,
        current_level: int
    ):
        """
        SCINet Tree.

        Parameters:
            in_planes: input channels
            hidden_size: hidden size for Conv1d
            kernel_size: kernel size for Conv1d
            groups: groups for Conv1d
            dropout: dropout ratio
            INN: whether to use interactive learning
            current_level: current level of tree
        """
        super(_SCINet_Tree, self).__init__()

        self.current_level = current_level

        self.workingblock = _LevelSCINet(
            in_planes=in_planes,
            hidden_size=hidden_size,
            kernel_size=kernel_size,
            groups=groups,
            dropout=dropout,
            INN=INN)

        if current_level != 0:
            self.SCINet_Tree_odd = _SCINet_Tree(
                in_planes,
                hidden_size,
                kernel_size,
                groups,
                dropout,
                INN,
                current_level-1)
            
            self.SCINet_Tree_even = _SCINet_Tree(
                in_planes, 
                hidden_size,
                kernel_size,
                groups,
                dropout,
                INN,
                current_level-1)
    
    def concat_and_realign(
        self,
        even: Tensor,
        odd: Tensor
    ) -> Tensor:
        """
        Concat & Realign.

        Parameters:
            even: even features
            odd: odd features

        Shape:
            even: (B, L, D)
            odd: (B, L, D)
            output: (B, L', D)
        """

        even = even.permute(1, 0, 2)    # (L, B, D)
        odd = odd.permute(1, 0, 2)      # (L, B, D)

        even_len = even.shape[0]
        odd_len = odd.shape[0]
        mlen = min((odd_len, even_len))

        concat = []
        for i in range(mlen):
            concat.append(even[i].unsqueeze(0))
            concat.append(odd[i].unsqueeze(0))
        if odd_len < even_len: 
            concat.append(even[-1].unsqueeze(0))

        output = torch.cat(concat, 0).permute(1,0,2)

        return  output
        
    def forward(
        self,
        x: Tensor
    ) -> Tensor:
        """
        Forward pass.

        Parameters:
            x: input features
        """
        F_even_update, F_odd_update = self.workingblock(x)

        if self.current_level == 0:
            return self.concat_and_realign(F_even_update, F_odd_update)
        else:
            return self.concat_and_realign(self.SCINet_Tree_even(F_even_update), self.SCINet_Tree_odd(F_odd_update))
        
class _EncoderTree(nn.Module):
    def __init__(
        self,
        in_planes: int,
        hidden_size: int,
        kernel_size: int,
        groups: int,
        dropout: float,
        INN: bool,
        num_levels: int
    ):
        """
        Encoder Tree.

        Parameters:
            in_planes: input channels
            hidden_size: hidden size
            kernel_size: kernel size
            groups: groups for Conv1d
            dropout: dropout ratio
            INN: whether to use interactive learning
            num_levels: number of level of the tree
        """
        super(_EncoderTree, self).__init__()

        self.levels = num_levels

        self.SCINet_Tree = _SCINet_Tree(
            in_planes=in_planes,
            hidden_size=hidden_size,
            kernel_size=kernel_size,
            groups=groups,
            dropout=dropout,
            INN=INN,
            current_level=num_levels-1)
        
    def forward(
        self,
        x: Tensor
    ) -> Tensor:
        """
        Forward pass.

        Parameters:
            x: input features

        Shape:
            x: (B, T, N)
            output: (B, L', D)
        """

        output = self.SCINet_Tree(x)

        return output
