"""
Baseline method, GWNet [IJCAI, 2019].
Author: ChunWei Shen

Reference:
* Paper: https://arxiv.org/abs/1906.00121
* Code: https://github.com/nnzhan/Graph-WaveNet
"""
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

class GWNet(nn.Module):
    """
    Graph Wavenet.

    Parameters:
        blocks: number of GWNet block
        layers: number of GWNet layers
        dropout: dropout ratio
        adpadj: whether to add self-adaptive adjacency matrix
        gcn_true: whether to add graph convolution layers
        n_adjs: number of transition matrices
        gcn_depth: depth of grpah convolution
        n_series: number of nodes
        t_window: lookback time window
        dilation_exponential: dilation exponential
        kernel_size: kernel size
        dilation_channels: dilation channels
        residual_channels: residual channels
        skip_channels: skip channels
        end_channels: end channels
        in_channels: input channels
        out_len: output sequence length
    """
    def __init__(
        self,
        st_params: Dict[str, Any],
        ch_params: Dict[str, Any],
        out_len: int
    ):
        super(GWNet, self).__init__()

        # Network parameters
        self.st_params = st_params
        self.ch_params = ch_params
        # hyperparameters of Spatial/Temporal Convolution Module
        self.blocks = self.st_params["blocks"]
        self.layers = self.st_params["layers"]
        dropout = self.st_params["dropout"]
        # Spatial
        self.adpadj = self.st_params["adpadj"]
        self.gcn_true = self.st_params["gcn_true"]
        n_adjs = self.st_params["n_adjs"] 
        gcn_depth = self.st_params["gcn_depth"]
        n_series  = self.st_params["n_series"]
        # Temporal
        self.t_window = self.st_params["t_window"]
        dilation_exponential = self.st_params["dilation_exponential"]
        kernel_size = self.st_params["kernel_size"]
        # hyperparameters of input/output channels
        dilation_channels = self.ch_params["dilation_channels"]
        residual_channels = self.ch_params["residual_channels"]
        skip_channels = self.ch_params["skip_channels"]
        end_channels = self.ch_params["end_channels"]
        in_channels = self.ch_params["in_channels"]

        self.gwnet_layers = nn.ModuleList()

        # linear layer for input transform
        self.start_conv = nn.Conv2d(
            in_channels = in_channels,
            out_channels = residual_channels,
            kernel_size = (1,1))

        receptive_field = 1
        
        if self.adpadj:
            self.sa = _SelfAdaptive(n_series)

        # Graph wavenet layers
        for b in range(self.blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(self.layers):
                self.gwnet_layers.append(
                    _GWNetLayer(
                        new_dilation,
                        kernel_size,
                        residual_channels,
                        dilation_channels,
                        skip_channels,
                        self.gcn_true,
                        gcn_depth,
                        dropout,
                        n_adjs
                    ))
                new_dilation *= dilation_exponential
                receptive_field += additional_scope
                additional_scope *= dilation_exponential
        
        # linear layer for the output of GWNet layer
        self.end_conv_1 = nn.Conv2d(
            in_channels=skip_channels,
            out_channels=end_channels,
            kernel_size=(1,1),
            bias=True)

        self.end_conv_2 = nn.Conv2d(
            in_channels=end_channels,
            out_channels=out_len,
            kernel_size=(1,1),
            bias=True)

        self.receptive_field = receptive_field

    def forward(
        self,
        input: Tensor,
        As: List[Tensor],
        **kwargs: Any,
    ) -> Tuple[Tensor, None, None]:
        """
        Forward pass.

        Parameters:
            x: input features
            As: list of adjacency matrices

        Return:
            output: prediction

        Shape:
            x: (B, P, N, C)
            As: each A with shape (N, N)
            output: (B, Q, N)
        """
        input = input.permute(0, 3, 2, 1)

        if self.t_window < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field - self.t_window,0,0,0))
        else:
            x = input

        # linear layer for input transform
        x = self.start_conv(x)
        skip = 0

        # Self Adaptive adjacency matrix
        if self.adpadj:
            if As is not None:
                adp = self.sa().to(x.device)
                As_aug = As + [adp]
            else:
                As_aug = [adp]

        # gwnet layers
        if self.adpadj:
            for gwnet in self.gwnet_layers:
                x, skip = gwnet(x, skip, As_aug)
        else:
            for gwnet in self.gwnet_layers:
                x, skip = gwnet(x, skip, As)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        output = torch.squeeze(x)

        return output, None, None

class _GWNetLayer(nn.Module):
    """
    Graph Wavenet layer.

    Parameters:
        new_dilation: dilated_factor in current layer
        kernel_size: size of kernel for convolution
        residual_channels: residual channels
        dilation_channels: dilation channels
        skip_channels: skip channels
        gcn_true: whether to add graph convolution layer
        gcn_depth: number of steps in diffusion process
        dropout: droupout ratio
        n_adjs: number of transition matrices
    """
    def __init__(
        self,
        new_dilation: int,
        kernel_size: int,
        residual_channels: int,
        dilation_channels: int,
        skip_channels: int,
        gcn_true: bool,
        gcn_depth: int,
        dropout: float,
        n_adjs: int,
    ):
        super(_GWNetLayer, self).__init__()

        self.gcn_true = gcn_true

        # dilated convolutions
        self.filter_convs = nn.Conv2d(
            in_channels = residual_channels,
            out_channels = dilation_channels,
            kernel_size = (1, kernel_size),
            dilation = new_dilation)

        self.gate_convs = nn.Conv2d(
            in_channels = residual_channels,
            out_channels = dilation_channels,
            kernel_size = (1, kernel_size), 
            dilation = new_dilation)
        
        # 1x1 convolution for residual connection
        self.residual_convs = nn.Conv2d(
            in_channels = dilation_channels,
            out_channels = residual_channels,
            kernel_size = (1, 1))

        # 1x1 convolution for skip connection
        self.skip_convs = nn.Conv2d(
            in_channels = dilation_channels,
            out_channels = skip_channels,
            kernel_size = (1, 1))
        
        self.bn = nn.BatchNorm2d(residual_channels)

        # Graph convolution
        if self.gcn_true:
            self.gconv = _GCN(
                dilation_channels,
                residual_channels,
                dropout,
                n_adjs,
                gcn_depth)
    
    def forward(
        self,
        x: Tensor,
        skip: Tensor,
        As: List[Tensor],
    ) -> Tensor:
        """
        Forward pass.

        Parameters:
            x: input feature
            x_skip: output for skip connection
            As: list of transition matrices

        Return:
            x: output 
            x_skip: output for skip connection

        Shape:
            x: (B, C, N, L)
            output: (B, C', N, L')
            x_skip: (B, C, N, L')
        """
        residual = x
        # dilated convolution
        filter = self.filter_convs(residual)
        filter = torch.tanh(filter)
        gate = self.gate_convs(residual)
        gate = torch.sigmoid(gate)
        x = filter * gate
        # skip connection
        s = self.skip_convs(x)
        try:
            skip = skip[:, :, :,  -s.size(3):]
        except:
            skip = 0
        skip = s + skip
        # graph convolution
        if self.gcn_true and As is not None:
            x = self.gconv(x, As)
        else:
            x = self.residual_convs(x)

        x = x + residual[:, :, :, -x.size(3):]
        output = self.bn(x)

        return output, skip

class _Linear(nn.Module):
    """
    Linear layer.

    Parameters:
        c_in: input channel number
        c_out: output channel number
        add_bias: whether to add bias
    """

    def __init__(
        self, 
        c_in: int, 
        c_out: int, 
        add_bias: bool = True
    ):
        super(_Linear, self).__init__()

        self.mlp = torch.nn.Conv2d(
            c_in,
            c_out,
            kernel_size = (1, 1),
            padding = (0, 0),
            stride = (1, 1),
            bias = add_bias)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Parameters:
            x: input features
        
        Return:
            output: the result after x passes through the mlp

        Shape:
            x: (B, c_in, N, L)
            output: (B, c_out, N, L)
        """

        output = self.mlp(x)

        return output

class _GCN(nn.Module):
    """
    Graph Convolution Layer.

    Parameters:
        c_in: number of input channel
        c_out: number of output channel
        dropout: dropout ratio
        n_adjs: number of transition matrices
        gcn_depth: number of steps in diffusion process
    """
    def __init__(
        self,
        c_in: int,
        c_out: int,
        dropout: float,
        n_adjs: int = 3,
        gcn_depth: int = 2
    ):
        super(_GCN,self).__init__()

        self.mlp = _Linear((gcn_depth * n_adjs + 1) * c_in, c_out)
        self.dropout = nn.Dropout(dropout)
        self.gcn_depth = gcn_depth

    def forward(
            self,
            x: Tensor,
            As: List[Tensor]
    ) -> Tensor:
        """
        Forward pass.

        Parameters:
            x: input features
            As: list of adjacency matrices
        
        Return:
            h: graph convolution output

        Shape:
            x: (B, c_in, N, L)
            h: (B, c_out, N, L)
        """
        out = [x]

        for A in As:
            for k in range(1, self.gcn_depth + 1):
                x = torch.einsum('ncvl,vw->ncwl',(x, A))
                out.append(x)

        h = torch.cat(out, dim = 1)   # Concat along channel
        h = self.mlp(h)
        h = self.dropout(h)

        return h

class _SelfAdaptive(nn.Module):
    """
    Construct self-adaptive adjacency matrix.
    
    n_series: number of nodes in the graph
    """
    def __init__(
        self,
        n_series: int
    ):
        super(_SelfAdaptive,self).__init__()

        self.nodevec1 = nn.Parameter(torch.randn(n_series, 10), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(10, n_series), requires_grad=True)
    
    def forward(self) -> Tensor:
        """
        Forward pass.

        Return:
            self_adaptive: self-adaptive adjacency matrix
        
        Shape:
            self_adaptive: (N, N)
        """

        self_adaptive = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)

        return self_adaptive