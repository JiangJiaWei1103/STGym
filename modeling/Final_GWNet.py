"""
Graph Wavenet framework.

Reference: 
https://github.com/nnzhan/Graph-WaveNet

Author: ChunWei Shen
"""

from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

class GWNet(nn.Module):
    """
    Graph Wavenet.

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
        out_dim: int,
        device: str,
        priori_gs: Optional[List[Tensor]] = None,
    ):
        super(GWNet, self).__init__()

        # Network parameters
        self.st_params = st_params
        self.ch_params = ch_params
        self.supports = priori_gs

        # hyperparameters of Spatial/Temporal Convolution Module
        self.blocks = self.st_params["blocks"]
        self.layers = self.st_params["layers"]
        dropout = self.st_params["dropout"]
        # Spatial
        self.addaptadj = self.st_params["addaptadj"]
        self.gcn_true = self.st_params["gcn_true"] 
        adaptive_init = self.st_params["adaptive_init"]
        randomadj = self.st_params["randomadj"]
        adaptive_only = self.st_params["adaptive_only"]
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

        self.start_conv = nn.Conv2d(
            in_channels = in_channels,
            out_channels = residual_channels,
            kernel_size = (1,1))

        receptive_field = 1

        if adaptive_only:
            self.supports = None
        else:
            self.supports = [torch.tensor(i).to(device) for i in self.supports]

        self.supports_len = 0
        if self.supports is not None:
            self.supports_len += len(self.supports)
        
        if not randomadj:
            adaptive_init = self.supports[0]
        if self.gcn_true and self.addaptadj:
            self.sa = _SelfAdaptive(adaptive_init, n_series)
            if self.supports is None:
                self.supports = []
            self.supports_len +=1

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
                        self.supports_len
                    ))
                new_dilation *= dilation_exponential
                receptive_field += additional_scope
                additional_scope *= dilation_exponential
        
        self.end_conv_1 = nn.Conv2d(
            in_channels=skip_channels,
            out_channels=end_channels,
            kernel_size=(1,1),
            bias=True)

        self.end_conv_2 = nn.Conv2d(
            in_channels=end_channels,
            out_channels=out_dim,
            kernel_size=(1,1),
            bias=True)

        self.receptive_field = receptive_field

        # self._reset_parameters()

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

        x = self.start_conv(x)
        skip = 0

        new_supports = None
        if self.gcn_true and self.addaptadj and self.supports is not None:
            adp = self.sa().to(x.device)
            new_supports = self.supports + [adp]

        # gwnet layers
        if self.addaptadj:
            for gwnet in self.gwnet_layers:
                x, skip = gwnet(x, skip, new_supports)
        else:
            for gwnet in self.gwnet_layers:
                x, skip = gwnet(x, skip, self.supports)

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
        supports_len: number of transition/adjacency matrices
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
        supports_len: int,
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

        if self.gcn_true:
            self.gconv = _GCN(
                dilation_channels,
                residual_channels,
                dropout,
                supports_len,
                gcn_depth)

        # self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)
    
    def forward(
        self,
        x: Tensor,
        skip: Tensor,
        supports: List[Tensor],
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
        if self.gcn_true and supports is not None:
            x = self.gconv(x, supports)
        else:
            x = self.residual_convs(x)

        x = x + residual[:, :, :, -x.size(3):]
        x = self.bn(x)

        return x, skip

class _Linear(nn.Module):
    """
    Linear layer.

    Parameters:
        c_in: input channel number
        c_out: output channel number
        add_bias:  whether to add bias
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

        # self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Parameters:
            x: input features
        
        Return:
            output: the result after x passes through the mlp

        Shape:
            x: (B, c_in, N, T)
            output: (B, c_out, N, T)
        """

        output = self.mlp(x)

        return output

class _GCN(nn.Module):
    """
    Graph Convolution Layer.

    Parameters:
        c_in: input channel number
        c_out: output channel number
        dropout:  dropout ratio
        support_len: number of transition/adjacency matrices
        gcn_depth: number of steps in diffusion process
    """
    def __init__(
        self,
        c_in: int,
        c_out: int,
        dropout: float,
        support_len: int = 3,
        gcn_depth: int = 2
    ):
        super(_GCN,self).__init__()

        self.mlp = _Linear((gcn_depth * support_len + 1) * c_in, c_out)
        self.dropout = dropout
        self.gcn_depth = gcn_depth

        # self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(
            self,
            x: Tensor,
            support: List[Tensor]
    ) -> Tensor:
        """
        Forward pass.

        Parameters:
            x: input features
            support: transition/adjacency matrices
        
        Return:
            h: final node embedding

        Shape:
            x: (B, c_in, N, T)
            h: (B, c_out, N, T)
        """
        out = [x]

        for A in support:
            x1 = torch.einsum('ncvl,vw->ncwl',(x,A))
            out.append(x1)
            for _ in range(2, self.gcn_depth + 1):
                x2 = torch.einsum('ncvl,vw->ncwl',(x1,A))
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim = 1)   # Concat along channel
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training = self.training)

        return h

class _SelfAdaptive(nn.Module):
    """
    construct self-adaptive adjacency matrix.
    
    adaptive_init: initialization of adaptive adjacency matrix
    num_nodes: number of nodes in the graph
    """
    def __init__(
        self,
        adaptive_init: Tensor,
        num_nodes: int
    ):
        super(_SelfAdaptive,self).__init__()

        if adaptive_init is None:
            self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10), requires_grad=True)
            self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes), requires_grad=True)
        else:
            m, p, n = torch.svd(adaptive_init)
            initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
            initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
            self.nodevec1 = nn.Parameter(initemb1, requires_grad=True)
            self.nodevec2 = nn.Parameter(initemb2, requires_grad=True)
        
        # self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)
    
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