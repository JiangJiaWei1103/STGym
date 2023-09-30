"""
ST-Norm framework.

Reference: 
https://github.com/nnzhan/Graph-WaveNet

Author: ChunWei Shen
"""
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

class STNorm(nn.Module):
    """
    ST-Norm.

    Parameters:
        st_params: hyperparameters of Spatial/Temporal Convolution Module
        ch_params: hyperparameters of input/output channels
        out_dim: output dimension
        device: device
        priori_gs: predefined adjacency matrix
    """
    def __init__(
        self,
        st_params: Dict[str, Any],
        ch_params: Dict[str, Any],
        stnorm_params: Dict[str, Any],
        out_dim: int,
        priori_gs: Optional[List[Tensor]] = None,
    ):
        super(STNorm, self).__init__()

        # Network parameters
        self.st_params = st_params
        self.ch_params = ch_params
        self.stnorm_params = stnorm_params

        # hyperparameters of Spatial/Temporal Convolution Module
        self.blocks = self.st_params["blocks"]
        self.layers = self.st_params["layers"]
        # Spatial
        n_series  = self.st_params["n_series"]
        # Temporal
        self.t_window = self.st_params["t_window"]
        dilation_exponential = self.st_params["dilation_exponential"]
        kernel_size = self.st_params["kernel_size"]
        
        # hyperparameters of input/output channels
        hid_channels = self.ch_params["hid_channels"]
        in_channels = self.ch_params["in_channels"]

        # hyperparameters of ST-Norm
        tnorm_bool = self.stnorm_params['tnorm_bool']
        snorm_bool = self.stnorm_params['snorm_bool']

        self.gwnet_layers = nn.ModuleList()

        # linear layer for input transform
        self.start_conv = nn.Conv2d(
            in_channels = in_channels,
            out_channels = hid_channels,
            kernel_size = (1,1))

        receptive_field = 1

        # Graph wavenet layers
        for b in range(self.blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(self.layers):
                self.gwnet_layers.append(
                    _WaveNetLayer(
                        new_dilation,
                        kernel_size,
                        hid_channels,
                        n_series,
                        tnorm_bool,
                        snorm_bool
                    ))
                new_dilation *= dilation_exponential
                receptive_field += additional_scope
                additional_scope *= dilation_exponential
        
        # linear layer for the output of GWNet layer
        self.end_conv_1 = nn.Conv2d(
            in_channels=hid_channels,
            out_channels=hid_channels,
            kernel_size=(1,1),
            bias=True)

        self.end_conv_2 = nn.Conv2d(
            in_channels=hid_channels,
            out_channels=out_dim,
            kernel_size=(1,1),
            bias=True)

        self.receptive_field = receptive_field

        self._reset_parameters()
         
    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self,
        input: Tensor,
        **kwargs: Any,
    ) -> Tuple[Tensor, None, None]:
        """
        Forward pass.

        Parameters:
            x: node feature matrix

        Return:
            output: prediction

        Shape:
            x: (B, T, N, C), where B is the batch_size, T is the lookback
                time window and N is the number of time series
            output: (B, out_dim, N)
        """
        input = input.permute(0, 3, 2, 1)

        if self.t_window < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field - self.t_window,0,0,0))
        else:
            x = input

        # linear layer for input transform
        x = self.start_conv(x)
        skip = 0

        for gwnet in self.gwnet_layers:
            x, skip = gwnet(x, skip)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        output = torch.squeeze(x)

        return output, None, None

class _WaveNetLayer(nn.Module):
    """
    Graph Wavenet layer.

    Parameters:
        new_dilation: dilated_factor in current layer
        kernel_size: size of kernel for convolution
        hid_channels: hidden channels
        n_series: number of nodes
        tnorm_bool: whether to add temporal normalization
        snorm_bool: whether to add spatial normalization
    """
    def __init__(
        self,
        new_dilation: int,
        kernel_size: int,
        hid_channels: int,
        n_series: int,
        tnorm_bool: bool,
        snorm_bool: bool
    ):
        super(_WaveNetLayer, self).__init__()

        self.tnorm_bool = tnorm_bool
        self.snorm_bool = snorm_bool
        num = int(tnorm_bool) + int(snorm_bool) + 1

        # ST-Norm
        if tnorm_bool:
            self.tnorm = TNorm(n_series, hid_channels)
        if snorm_bool:
            self.snorm = SNorm(hid_channels)

        # dilated convolutions
        self.filter_convs = nn.Conv2d(
            in_channels = num * hid_channels,
            out_channels = hid_channels,
            kernel_size = (1, kernel_size),
            dilation = new_dilation)

        self.gate_convs = nn.Conv2d(
            in_channels = num * hid_channels,
            out_channels = hid_channels,
            kernel_size = (1, kernel_size), 
            dilation = new_dilation)
        
        # 1x1 convolution for residual connection
        self.residual_convs = nn.Conv2d(
            in_channels = hid_channels,
            out_channels = hid_channels,
            kernel_size = (1, 1))

        # 1x1 convolution for skip connection
        self.skip_convs = nn.Conv2d(
            in_channels = hid_channels,
            out_channels = hid_channels,
            kernel_size = (1, 1))
    
    def forward(
        self,
        x: Tensor,
        skip: Tensor,
    ) -> Tensor:
        """
        Forward pass.

        Parameters:
            x: node feature matrix
            x_skip: node feature matrix for skip connection
            supports: transition/adjacency matrices

        Return:
            x: output 
            x_skip: output for skip connection

        Shape:
            x: (B, C, N, T)
            x_skip: (B, C, N, T)
        """
        residual = x
        x_list = []
        x_list.append(x)
        # ST-Norm
        if self.tnorm_bool:
            x_tnorm = self.tnorm(x)
            x_list.append(x_tnorm)    
        if self.snorm_bool:
            x_snorm = self.snorm(x)
            x_list.append(x_snorm)
        
        x = torch.cat(x_list, dim = 1)
        
        # dilated convolution
        filter = self.filter_convs(x)
        filter = torch.tanh(filter)
        gate = self.gate_convs(x)
        gate = torch.sigmoid(gate)
        x = filter * gate
        # skip connection
        s = self.skip_convs(x)
        try:
            skip = skip[:, :, :,  -s.size(3):]
        except:
            skip = 0
        skip = s + skip

        x = x + residual[:, :, :, -x.size(3):]

        return x, skip

class SNorm(nn.Module):
    """
    Spatial Normalization.

    Parameters:
        channels: input channels
    """
    def __init__(
        self,
        channels: int
    ):
        super(SNorm, self).__init__()

        self.beta = nn.Parameter(torch.zeros(channels))
        self.gamma = nn.Parameter(torch.ones(channels))

    def forward(
        self,
        x: Tensor
    ) -> Tensor:
        x_norm = (x - x.mean(2, keepdims=True)) / (x.var(2, keepdims=True, unbiased=True) + 0.00001) ** 0.5
        out = x_norm * self.gamma.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)

        return out

class TNorm(nn.Module):
    """
    Temporal Normalization.

    Parameters:
        n_series: number of nodes
        channels: input channels
        track_running_stats: whether to track running stats
        momentum: momentum
    """
    def __init__(
        self,
        n_series: int,
        channels: int,
        track_running_stats:bool = True,
        momentum:float = 0.1
    ):
        super(TNorm, self).__init__()

        self.track_running_stats = track_running_stats
        self.beta = nn.Parameter(torch.zeros(1, channels, n_series, 1))
        self.gamma = nn.Parameter(torch.ones(1, channels, n_series, 1))
        self.register_buffer('running_mean', torch.zeros(1, channels, n_series, 1))
        self.register_buffer('running_var', torch.ones(1, channels, n_series, 1))
        self.momentum = momentum

    def forward(
        self,
        x: Tensor
    ) -> Tensor:
        if self.track_running_stats:
            mean = x.mean((0, 3), keepdims=True)
            var = x.var((0, 3), keepdims=True, unbiased=False)
            if self.training:
                n = x.shape[3] * x.shape[0]
                with torch.no_grad():
                    self.running_mean = self.momentum * mean + (1 - self.momentum) * self.running_mean
                    self.running_var = self.momentum * var * n / (n - 1) + (1 - self.momentum) * self.running_var
            else:
                mean = self.running_mean
                var = self.running_var
        else:
            mean = x.mean((3), keepdims=True)
            var = x.var((3), keepdims=True, unbiased=True)
        x_norm = (x - mean) / (var + 0.00001) ** 0.5
        out = x_norm * self.gamma + self.beta

        return out