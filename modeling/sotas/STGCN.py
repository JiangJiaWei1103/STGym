"""
Baseline method, STGCN [IJCAI, 2018].
Author: ChunWei Shen

Reference:
* Paper: https://arxiv.org/abs/1709.04875
* Code: https://github.com/hazdzz/STGCN
"""
import math
from typing import List, Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.init as init
import torch.nn.functional as F


class STGCN(nn.Module):
    """
    STGCN.

    Parameters:
        Kt: kernel size of temporal convolution layers
        Ks: order of Chebyshev Polynomials Approximation
        blocks: input/output dimension of STGCN blocks
        n_series: number of nodes
        t_window: lookback time windows
        act_func: activation function in temporal convolution layers
        graph_conv_type: type of graph convolution (cheb_graph_conv or graph_conv)
        bias: whether to add bias
        dropout: dropout ratio
    """
    def __init__(
        self,
        st_params: Dict[str, Any],
    ):
        super(STGCN, self).__init__()

        # Network parameters
        self.st_params = st_params

        # hyperparameters of Spatial/Temporal pattern extractor
        Kt = self.st_params['Kt']
        Ks = self.st_params['Ks']
        self.blocks = self.st_params['blocks']
        n_series = self.st_params['n_series']
        t_window = self.st_params['t_window']
        act_func = self.st_params['act_func']
        graph_conv_type = self.st_params['graph_conv_type']
        enable_bias = self.st_params['bias']
        droprate = self.st_params['dropout']

        # Spatio-temporal Convolution Block
        self.st_blocks = nn.ModuleList()
        for l in range(len(self.blocks) - 3):
            self.st_blocks.append(
                _STConvBlock(
                    Kt,
                    Ks,
                    n_series,
                    self.blocks[l][-1],
                    self.blocks[l+1],
                    act_func,
                    graph_conv_type,
                    enable_bias,
                    droprate))

        # output layer
        Ko = t_window - (len(self.blocks) - 3) * 2 * (Kt - 1)
        self.Ko = Ko
        if self.Ko > 1:
            self.output = _OutputBlock(
                Ko,
                self.blocks[-3][-1],
                self.blocks[-2],
                self.blocks[-1][0],
                n_series,
                act_func,
                enable_bias,
                droprate)
        elif self.Ko == 0:
            self.fc1 = nn.Linear(
                in_features=self.blocks[-3][-1],
                out_features=self.blocks[-2][0],
                bias=enable_bias)
            self.fc2 = nn.Linear(
                in_features=self.blocks[-2][0],
                out_features=self.blocks[-1][0],
                bias=enable_bias)
            self.relu = nn.ReLU()
            self.leaky_relu = nn.LeakyReLU()
            self.silu = nn.SiLU()
            self.dropout = nn.Dropout(p=droprate)

    def forward(
        self, 
        x: Tensor,
        As: List[Tensor],
        **kwargs: Any,
    ) -> Tuple[Tensor, None, None]:
        """
        Forward pass.

        Parameters:
            input: input features
            As: list of adjacency matrices

        Shape:
            input: (B, P, N, C)
            output: (B, Q, N)
        """

        x = x.permute(0, 3, 1, 2)               # (B, C, P, N)

        # Spatio-temporal Convolution Block
        for i in range(len(self.blocks) - 3):
            x = self.st_blocks[i](x, As)

        # output layer
        if self.Ko > 1:
            x = self.output(x)
        elif self.Ko == 0:
            x = self.fc1(x.permute(0, 2, 3, 1)) # (B, L, N, C')
            x = self.relu(x)
            x = self.fc2(x).permute(0, 3, 1, 2) # (B, C', L, N)

        x = x.squeeze(2)
        
        return x, None, None

class _Align(nn.Module):
    """
    Ensure alignment of input feature dimensions for the residual connection.

    Parameters:
        c_in: input channels
        c_out: output channels
    """
    def __init__(
        self, 
        c_in: int, 
        c_out: int
    ):
        super(_Align, self).__init__()

        self.c_in = c_in
        self.c_out = c_out
        self.align_conv = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=(1, 1))

    def forward(
        self, 
        x: Tensor
    ) -> Tensor:
        """
        Forward pass.

        Parameters:
            x: input features
        
        Shape:
            x: (B, C, L, N)
        """
        if self.c_in > self.c_out:
            x = self.align_conv(x)
        elif self.c_in < self.c_out:
            batch_size, _, t_window, n_series = x.shape
            zeros = torch.zeros([batch_size, self.c_out - self.c_in, t_window, n_series]).to(x.device)
            x = torch.cat([x, zeros], dim = 1)
        else:
            x = x
        
        return x
    
class _CausalConv2d(nn.Conv2d):
    """
    2-D causal convolution.

    Parameters:
        in_channels: input channels
        out_channels: output channels
        kernel_size: kernel size
        stride: stride
        enable_padding:  whether to padding or not
        dilation: dilation factor
        groups: controls the connections between inputs and outputs
        bias: whether to add bias or not
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        enable_padding: bool = False,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True
    ):
        kernel_size = nn.modules.utils._pair(kernel_size)
        stride = nn.modules.utils._pair(stride)
        dilation = nn.modules.utils._pair(dilation)
        
        if enable_padding == True:
            self.__padding = [int((kernel_size[i] - 1) * dilation[i]) for i in range(len(kernel_size))]
        else:
            self.__padding = 0

        self.left_padding = nn.modules.utils._pair(self.__padding)

        super(_CausalConv2d, self).__init__(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=stride, 
            padding=0, 
            dilation=dilation, 
            groups=groups, 
            bias=bias)
        
    def forward(
        self, 
        input: Tensor
    ):
        """
        Forward pass.

        Parameters:
            input: input features
        
        Shape:
            input: (B, C, L, N)
            result: (B, C', L, N)
        """
        if self.__padding != 0:
            input = F.pad(input, (self.left_padding[1], 0, self.left_padding[0], 0))

        result = super(_CausalConv2d, self).forward(input)

        return result

class _TemporalConvLayer(nn.Module):
    """
    Temporal Convolution Layer (GLU).

           |--------------------------------| * residual connection *
           |                                |
           |    |--->--- casualconv2d ----- + -------|       
    -------|----|                                   ⊙ ------>
                |--->--- casualconv2d --- sigmoid ---|

    Parameters:
        Kt: kernel size
        c_in: input channels
        c_out: output channels
        act_func: activation function
    """
    def __init__(
        self, 
        Kt: int,
        c_in: int,
        c_out: int,
        act_func: str
    ):
        super(_TemporalConvLayer, self).__init__()

        self.Kt = Kt
        self.c_out = c_out
        self.align = _Align(c_in, c_out)

        if act_func == 'glu' or act_func == 'gtu':
            self.causal_conv = _CausalConv2d(
                in_channels=c_in,
                out_channels=2 * c_out,
                kernel_size=(Kt, 1),
                enable_padding=False,
                dilation=1)
        else:
            self.causal_conv = _CausalConv2d(
                in_channels=c_in,
                out_channels=c_out,
                kernel_size=(Kt, 1),
                enable_padding=False,
                dilation=1)
            
        self.act_func = act_func
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.silu = nn.SiLU()

    def forward(
        self,
        x: Tensor
    ) -> Tensor:
        """
        Forward pass.

        Parameters:
            x: input features
        
        Shape:
            x: (B, C, L, N)
        """
        x_in = self.align(x)[:, :, self.Kt - 1:, :]
        x_causal_conv = self.causal_conv(x)

        if self.act_func == 'glu' or self.act_func == 'gtu':
            x_p = x_causal_conv[:, :self.c_out, :, :]
            x_q = x_causal_conv[:, -self.c_out:, :, :]

            if self.act_func == 'glu': 
                x = torch.mul((x_p + x_in), self.sigmoid(x_q))
            else:
                x = torch.mul(self.tanh(x_p + x_in), self.sigmoid(x_q))
        elif self.act_func == 'relu':
            x = self.relu(x_causal_conv + x_in)
        elif self.act_func == 'leaky_relu':
            x = self.leaky_relu(x_causal_conv + x_in)
        elif self.act_func == 'silu':
            x = self.silu(x_causal_conv + x_in)
        else:
            raise NotImplementedError(f'ERROR: The activation function {self.act_func} is not implemented.')
        
        return x
    
class _ChebGraphConv(nn.Module):
    """
    Chebyshev graph convolution.

    Parameters:
        c_in: input channels
        c_out: output channels
        Ks: order of Chebyshev Polynomials Approximation
        bias: whether to add bias
    """
    def __init__(
        self,
        c_in: int,
        c_out: int,
        Ks: int,
        bias: bool
    ):
        super(_ChebGraphConv, self).__init__()

        self.c_in = c_in
        self.c_out = c_out
        self.Ks = Ks
        self.weight = nn.Parameter(torch.FloatTensor(Ks, c_in, c_out))

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(c_out))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a = math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
    
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
        
        Shape:
            x: (B, C, L, N)
            cheb_graph_conv: (B, L, N, C')
        """
        x = torch.permute(x, (0, 2, 3, 1))  # (B, L, N, C)

        if self.Ks - 1 < 0:
            raise ValueError(f'ERROR: the graph convolution kernel size Ks \
                             has to be a positive integer, but received {self.Ks}.')
        elif self.Ks - 1 == 0:
            x_0 = x
            x_list = [x_0]
        elif self.Ks - 1 == 1:
            x_0 = x
            x_1 = torch.einsum('hi,btij->bthj', As[0], x)
            x_list = [x_0, x_1]
        elif self.Ks - 1 >= 2:
            x_0 = x
            x_1 = torch.einsum('hi,btij->bthj', As[0], x)
            x_list = [x_0, x_1]
            for k in range(2, self.Ks):
                x_list.append(torch.einsum('hi,btij->bthj', 2 * As[0], x_list[k - 1]) - x_list[k - 2])
        
        x = torch.stack(x_list, dim = 2)
        cheb_graph_conv = torch.einsum('btkhi,kij->bthj', x, self.weight)

        if self.bias is not None:
            cheb_graph_conv = torch.add(cheb_graph_conv, self.bias)
        
        return cheb_graph_conv

class _GraphConv(nn.Module):
    """
    Graph convolution.

    Parameters:
        c_in: input channels
        c_out: output channels
        bias: whether to add bias
    """
    def __init__(
        self,
        c_in: int,
        c_out: int,
        bias: bool
    ):
        super(_GraphConv, self).__init__()

        self.c_in = c_in
        self.c_out = c_out

        self.weight = nn.Parameter(torch.FloatTensor(c_in, c_out))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(c_out))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a = math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

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
        
        Shape:
            x: (B, C, L, N)
            graph_conv: (B, L, N, C')
        """
        x = torch.permute(x, (0, 2, 3, 1))  # (B, L, N, C)

        first_mul = torch.einsum('hi,btij->bthj', As[0], x)
        second_mul = torch.einsum('bthi,ij->bthj', first_mul, self.weight)

        if self.bias is not None:
            graph_conv = torch.add(second_mul, self.bias)
        else:
            graph_conv = second_mul
        
        return graph_conv

class _GraphConvLayer(nn.Module):
    """
    Graph convolution layers.

    Parameters:
        graph_conv_type: graph convolution type (cheb_graph_conv or graph_conv)
        c_in: input channels
        c_out: output channels
        Ks: order of Chebyshev Polynomials Approximation
        bias: whether to add bias
    """
    def __init__(
        self,
        graph_conv_type: str,
        c_in: int,
        c_out: int,
        Ks: int,
        bias: bool
    ):
        super(_GraphConvLayer, self).__init__()

        self.graph_conv_type = graph_conv_type
        self.c_in = c_in
        self.c_out = c_out
        self.align = _Align(c_in, c_out)
        self.Ks = Ks

        if self.graph_conv_type == 'cheb_graph_conv':
            self.cheb_graph_conv = _ChebGraphConv(c_out, c_out, Ks, bias)
        elif self.graph_conv_type == 'graph_conv':
            self.graph_conv = _GraphConv(c_out, c_out, bias)

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

        Shape:
            x: (B, C, L, N)
            x_gc_out: (B, C', L, N)
        """
        x_gc_in = self.align(x)

        if self.graph_conv_type == 'cheb_graph_conv':
            x_gc = self.cheb_graph_conv(x_gc_in, As)
        elif self.graph_conv_type == 'graph_conv':
            x_gc = self.graph_conv(x_gc_in, As)

        x_gc = x_gc.permute(0, 3, 1, 2)     # (B, C', L, N)
        x_gc_out = torch.add(x_gc, x_gc_in) # residual connection

        return x_gc_out
    
class _STConvBlock(nn.Module):
    """
    STConv Block contains 'TGTND' structure.
    T: Gated Temporal Convolution Layer (GLU or GTU)
    G: Graph Convolution Layer (ChebGraphConv or GraphConv)
    T: Gated Temporal Convolution Layer (GLU or GTU)
    N: Layer Normolization
    D: Dropout

    Parameters:
        Kt: kernel size
        Ks: order of Chebyshev Polynomials Approximation
        n_series: number of nodes
        last_block_channel: last block channel
        channels: number of channels in the Temporal/Graph Convolution Layer
        act_func: activation function
        graph_conv_type: graph convolution type (cheb_graph_conv or graph_conv)
        bias: whether to add bias
        droprate: dropout rate
    """
    def __init__(
        self,
        Kt: int,
        Ks: int,
        n_series: int,
        last_block_channel: int,
        channels: List[List[int]],
        act_func: str,
        graph_conv_type: str,
        bias: bool,
        droprate: float
    ):
        super(_STConvBlock, self).__init__()

        self.tmp_conv1 = _TemporalConvLayer(Kt, last_block_channel, channels[0], act_func)
        self.graph_conv = _GraphConvLayer(graph_conv_type, channels[0], channels[1], Ks, bias)
        self.tmp_conv2 = _TemporalConvLayer(Kt, channels[1], channels[2], act_func)
        self.tc2_ln = nn.LayerNorm([n_series, channels[2]])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=droprate)

    def forward(
        self,
        x: Tensor,
        As: List[Tensor],
    ) -> Tensor:
        """
        Forward pass.

        Parameters:
            x: input features
            As: list of adjacency matrices

        Shape:
            x: (B, C, L, N)
        """
        x = self.tmp_conv1(x)
        x = self.graph_conv(x, As)
        x = self.relu(x)
        x = self.tmp_conv2(x)
        x = self.tc2_ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # (B, L, N, C') -> (B, C', L, N)
        x = self.dropout(x)

        return x

class _OutputBlock(nn.Module):
    """
    Output block contains 'TNFF' structure.
    T: Gated Temporal Convolution Layer (GLU or GTU)
    N: Layer Normolization
    F: Fully-Connected Layer
    F: Fully-Connected Layer

    Parameters:
        Ko: kernel size
        last_block_channel: last block channel
        channels: number of channels in the Temporal/Graph Convolution Layer
        end_channel: end channel
        n_series: number of nodes
        act_func: activation function
        bias: whether to add bias
        droprate: dropout rate
    """
    def __init__(
        self,
        Ko: int,
        last_block_channel: int,
        channels: List[List[int]],
        end_channel: int,
        n_series: int,
        act_func: str,
        bias: bool,
        droprate: float
    ):
        super(_OutputBlock, self).__init__()

        self.tmp_conv1 = _TemporalConvLayer(Ko, last_block_channel, channels[0], act_func)
        self.fc1 = nn.Linear(
            in_features=channels[0],
            out_features=channels[1],
            bias=bias)
        self.fc2 = nn.Linear(
            in_features=channels[1],
            out_features=end_channel,
            bias=bias)
        self.tc1_ln = nn.LayerNorm([n_series, channels[0]])
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.silu = nn.SiLU()
        self.dropout = nn.Dropout(p=droprate)

    def forward(
        self,
        x: Tensor
    ) -> Tensor:
        """
        Forward pass.

        Parameters:
            x: input features

        Shape:
            x: (B, C, L, N)
        """
        x = self.tmp_conv1(x)
        x = self.tc1_ln(x.permute(0, 2, 3, 1))  # (B, L, N, C')
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x).permute(0, 3, 1, 2)     # (B, C', L, N)

        return x