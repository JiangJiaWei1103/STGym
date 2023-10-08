"""
MTGNN framework.

Reference: 
https://github.com/nnzhan/MTGNN
https://github.com/benedekrozemberczki/pytorch_geometric_temporal

Author: ChunWei Shen
"""
from typing import List, Dict, Any, Optional, Union, Tuple

import torch
import torch.nn as nn
from torch.nn import init
from torch import Tensor
import torch.nn.functional as F

class MTGNN(nn.Module):
    """
    MTGNN.

    Parameters:
        st_params: hyperparameters of Spatial/Temporal Convolution Module
        gl_params: hyperparameters of Graph Learning Layer
        ch_params: hyperparameters of input/output channels
        out_dim: output dimension
        priori_gs: predefined adjacency matrix
    """

    def __init__(
        self,
        st_params: Dict[str, Any],
        gl_params: Dict[str, Any],
        ch_params: Dict[str, Any],
        out_dim: int,
        priori_gs: Optional[List[Tensor]] = None
    ):
        super(MTGNN, self).__init__()

        # Network parameters
        self.st_params = st_params
        self.gl_params = gl_params
        self.ch_params = ch_params
        self.priori_gs = priori_gs

        # hyperparameters of Spatial/Temporal Convolution Module
        self.layers = self.st_params["layers"]
        self.dropout = self.st_params["dropout"]
        # Spatial
        self.gcn_true = self.st_params["gcn_true"]
        gcn_depth = self.st_params["gcn_depth"]
        n_series = self.st_params["n_series"]
        propalpha = self.st_params["propalpha"]
        # Temporal
        self.t_window = self.st_params["t_window"]
        dilation_exponential = self.st_params["dilation_exponential"]
        kernel_set = self.st_params["kernel_set"]
        kernel_size = kernel_set[-1]

        # hyperparameters of Graph Learning Layer
        self.idx = torch.arange(n_series)
        self.buildA_true = self.gl_params["buildA_true"]
        static_feature = self.gl_params["static_feature"]
        subgraph_size = self.gl_params["subgraph_size"]
        node_dim = self.gl_params["node_dim"]
        tanhalpha = self.gl_params["tanhalpha"]

        # hyperparameters of input/output channels
        conv_channels = self.ch_params["conv_channels"]
        residual_channels = self.ch_params["residual_channels"]
        skip_channels = self.ch_params["skip_channels"]
        end_channels = self.ch_params["end_channels"]
        in_channels = self.ch_params["in_channels"]

        # hyperparameters of Layer Normalization
        layer_norm_affline = self.st_params["layer_norm_affline"]


        self.mtgnn_layers = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels = in_channels,
                                    out_channels = residual_channels,
                                    kernel_size = (1, 1))
                                    
        self.gl = _GraphLearning(n_series, 
                                 subgraph_size, 
                                 node_dim, 
                                 alpha = tanhalpha, 
                                 static_feature = static_feature)
        
        if dilation_exponential > 1:
            self.receptive_field = int(
                1
                +(kernel_size - 1)
                *(dilation_exponential**self.layers - 1)
                /(dilation_exponential - 1))
        else:
            self.receptive_field = self.layers * (kernel_size - 1) + 1


        new_dilation = 1
        for j in range(1,self.layers+1):
            self.mtgnn_layers.append(
                _MTGNNLayer(
                    dilation_exponential = dilation_exponential,
                    j = j,
                    residual_channels = residual_channels,
                    conv_channels = conv_channels,
                    skip_channels = skip_channels,
                    kernel_size = kernel_size,
                    kernel_set = kernel_set,
                    new_dilation = new_dilation,
                    layer_norm_affline = layer_norm_affline,
                    gcn_true = self.gcn_true,
                    t_window = self.t_window,
                    receptive_field = self.receptive_field,
                    dropout = self.dropout,
                    gcn_depth = gcn_depth,
                    n_series = n_series,
                    propalpha = propalpha,
                )
            )
            
            new_dilation *= dilation_exponential

        
        self.end_conv_1 = nn.Conv2d(in_channels = skip_channels,
                                    out_channels = end_channels,
                                    kernel_size = (1,1),
                                    bias = True)
        self.end_conv_2 = nn.Conv2d(in_channels = end_channels,
                                    out_channels = out_dim,
                                    kernel_size = (1,1),
                                    bias = True)
        
        # First and last Skip connections
        if self.t_window > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels = in_channels, 
                                   out_channels = skip_channels, 
                                   kernel_size = (1, self.t_window), 
                                   bias = True)
            self.skipE = nn.Conv2d(in_channels = residual_channels, 
                                   out_channels = skip_channels, 
                                   kernel_size = (1, self.t_window-self.receptive_field+1), 
                                   bias = True)
        else:
            self.skip0 = nn.Conv2d(in_channels = in_channels, 
                                   out_channels = skip_channels, 
                                   kernel_size = (1, self.receptive_field), 
                                   bias = True)
            self.skipE = nn.Conv2d(in_channels = residual_channels, 
                                   out_channels = skip_channels, 
                                   kernel_size = (1, 1),
                                   bias = True)
    
    def forward(self,
        input: Tensor,
        idx: Optional[Tensor] = None,
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
        input = input.permute(0, 3, 2, 1)   # (B, C, N, T)
        
        seq_len = input.size(3)
        assert seq_len==self.t_window, 'input sequence length not equal to preset t_window'

        if self.t_window < self.receptive_field:
            input = nn.functional.pad(input,(self.receptive_field-self.t_window,0,0,0))
        
        if self.gcn_true:
            if self.buildA_true:
                if idx is None:
                    A = self.gl(self.idx.to(input.device))
                else:
                    A = self.gl(idx)
            else:
                A = self.priori_gs
            
        x = self.start_conv(input)
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))
        # mtgnn layers
        if idx is None:
            for mtgnn in self.mtgnn_layers:
                x, skip = mtgnn(x, skip, A, self.idx.to(x.device), self.training)
        else:
            for mtgnn in self.mtgnn_layers:
                x, skip = mtgnn(x, skip, A, idx, self.training)
        
        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        output = x.squeeze(-1).squeeze(1)

        return output, None, None

class _MTGNNLayer(nn.Module):
    """
    MTGNN layer.

    Parameters:
        dilation_exponential: dilation exponential
        j: iteration index, current layer of MTGNN
        residual_channels: residual channels
        conv_channels: convolution channels
        skip_channels: skip channels
        kernel_size: size of kernel for convolution
        kernel_set : list of kernel sizes
        new_dilation: dilated_factor in current layer
        layer_norm_affline: whether to do elementwise affine in Layer Normalization
        gcn_true: whether to add graph convolution layer
        t_window: lookback time window
        receptive_field: receptive field of the network
        dropout: droupout ratio
        gcn_depth: depth of graph convolution
        n_series: number of time series
        propalpha: retaining ratio of the original state of node features in mix-hop propagation
    """

    def __init__(
        self,
        dilation_exponential: int,
        j: int,
        residual_channels: int,
        conv_channels: int,
        skip_channels: int,
        kernel_size: int,
        kernel_set: list,
        new_dilation: int,
        layer_norm_affline: bool,
        gcn_true: bool,
        t_window: int,
        receptive_field: int,
        dropout: float,
        gcn_depth: int,
        n_series: int,
        propalpha: float,
    ):
        super(_MTGNNLayer, self).__init__()

        self.dropout = dropout
        self.gcn_true = gcn_true

        # rf_size_j: size of receptive field in layer j
        if dilation_exponential > 1:
            rf_size_j = int(
                1 
                + (kernel_size - 1) 
                * (dilation_exponential ** j - 1) 
                / (dilation_exponential - 1))
        else:
            rf_size_j = 1 + j * (kernel_size - 1)

        # TC Module, Dilated Inception Layer
        self.filter_convs = _DilatedInception(
            residual_channels,
            conv_channels,
            kernel_set = kernel_set,
            dilated_factor = new_dilation)
        
        self.gate_convs =  _DilatedInception(
            residual_channels,
            conv_channels,
            kernel_set = kernel_set,
            dilated_factor = new_dilation)
        
        self.residual_convs = nn.Conv2d(
            in_channels = conv_channels,
            out_channels = residual_channels,
            kernel_size = (1, 1))

        # Skip connections
        if t_window > receptive_field:
            self.skip_convs = nn.Conv2d(
                in_channels = conv_channels,
                out_channels = skip_channels,
                kernel_size = (1, t_window-rf_size_j+1))
        else:
            self.skip_convs = nn.Conv2d(
                in_channels = conv_channels,
                out_channels = skip_channels,
                kernel_size = (1, receptive_field-rf_size_j+1))
        
        # GC Module, Mix-hop Propagation Layer
        if gcn_true:
            self.gconv1 = _MixProp(
                conv_channels,
                residual_channels,
                gcn_depth, 
                dropout,
                propalpha)
            self.gconv2 =  _MixProp(
                conv_channels,
                residual_channels,
                gcn_depth,
                dropout,
                propalpha)
        
        # Layer Normalization Layer
        if t_window > receptive_field:
            self.norm = _LayerNormalization(
                (residual_channels, n_series, t_window - rf_size_j + 1),
                elementwise_affine = layer_norm_affline)
        else:
            self.norm = _LayerNormalization(
                (residual_channels, n_series, receptive_field - rf_size_j + 1),
                elementwise_affine = layer_norm_affline)
    
    def forward(
        self,
        x: Tensor,
        skip: Tensor,
        A: Optional[Tensor],
        idx: Tensor,
        training: bool,
    ) -> Tensor:
        """
        Forward pass.

        Parameters:
            x: node feature matrix
            x_skip: node feature matrix for skip connection
            A: adjacency matrix
            idx: input indices, a permutation of the number of nodes
            training: whether in traning mode

        Return:
            x: output 
            x_skip: output for skip connection

        Shape:
            x: (B, C, N, T)
            x_skip: (B, C, N, T)
            A: (N, N)
        """
        residual = x
        # TC Module, Dilated Inception Layer
        filter = self.filter_convs(x)
        filter = torch.tanh(filter)
        gate = self.gate_convs(x)
        gate = torch.sigmoid(gate)
        x = filter * gate
        x = F.dropout(x, self.dropout, training=training)
        # Skip connections
        skip = self.skip_convs(x) + skip
        # GC Module, Mix-hop Propagation Layer
        if self.gcn_true:
            x = self.gconv1(x, A) + self.gconv2(x, A.transpose(1,0))
        else:
            x = self.residual_convs(x)

        x = x + residual[:, :, :, -x.size(3):]
        x = self.norm(x, idx)

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
    
    def forward(
        self,
        x: Tensor
    ) -> Tensor:
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

class _MixProp(nn.Module):
    """
    Mix-hop propagation layer.

    Parameters:
        c_in: input channel number
        c_out: output channel number
        gcn_depth: depth of graph convolution
        alpha: retaining ratio of the original state of node features
        dropout: dropout ratio
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        gcn_depth: int,
        alpha: float = 0.05,
        dropout: Optional[float] = None,
    ):
        self.name = self.__class__.__name__
        super(_MixProp, self).__init__()

        # Network parameters
        self.c_in = c_in
        self.c_out = c_out
        self.gcn_depth = gcn_depth
        self.alpha = alpha

        self.mlp = _Linear((gcn_depth + 1) * c_in, c_out)

    def forward(
        self,
        x: Tensor,
        A: Tensor
    ) -> Tensor:
        """
        Forward pass.

        Parameters:
            x: input features
            A: adjacency matrix

        Return:
            h: final node embedding

        Shape:
            x: (B, C, N, T)
            A: (N, N)
            h: (B, C, N, T)
        """

        assert x.dim() == 4, "Shape of node features doesn't match (B, C, N, T)."

        A = A + torch.eye(A.size(0)).to(x.device)
        D = A.sum(1)
        A = A / D.view(-1, 1)

        # Information propagation
        h_0 = x
        h = x
        for hop in range(self.gcn_depth):
            h = self.alpha * x + (1 - self.alpha) * torch.einsum(
                "ncwl,vw->ncvl", (h, A)  # (B, C, N, T), (N, N)
            )  # (B, C, N, T)
            h_0 = torch.cat((h_0, h), dim = 1)  # Concat along channel

        # Information selection
        h = self.mlp(h_0)

        return h

class _DilatedInception(nn.Module):
    """
    Dilated inception layer.

    Parameters:
        c_in: input channel number
        c_out: output channel number
        kernel_set : list of kernel sizes
        dilated_factor : dilation factor
    """

    def __init__(
        self, 
        c_in: int, 
        c_out: int, 
        kernel_set: List[int] = [2, 3, 6, 7], 
        dilated_factor: int = 2
    ):
        super(_DilatedInception, self).__init__()

        self.tconv = nn.ModuleList()
        self.kernel_set = kernel_set
        c_out = int(c_out/len(kernel_set))

        for kernel_size in self.kernel_set:
            self.tconv.append(
                nn.Conv2d(
                    c_in,
                    c_out,
                    kernel_size = (1, kernel_size),
                    dilation = (1, dilated_factor)))

    def forward(
        self,
        x: Tensor
    ) -> Tensor:
        """
        Forward pass.

        Parameters:
            x: input features
        
        Return:
            z: final node embedding

        Shape:
            x: (B, c_in, N, T)
            z: (B, c_out, N, T')
        """

        z = []
        for i in range(len(self.kernel_set)):
            z.append(self.tconv[i](x))

        # truncated to the same length 
        # according to the largest filter
        for i in range(len(self.kernel_set)):
            z[i] = z[i][..., -z[-1].size(3):]
        
        z = torch.cat(z, dim = 1) # Concat along channel

        return z

class _GraphLearning(nn.Module):
    """
    Graph learning layer.
    Construct an adjacency matrix from node embeddings.

    Parameters:
        num_nodes: number of nodes in the graph
        k: number of top closest neighbors
        embedding_dim: dimension of node embedding
        alpha: controlling the saturation rate of the activation function
        static_feature: external knowledge about the attributes of each node
    """
    
    def __init__(
        self, 
        num_nodes: int,
        k: int,
        embedding_dim: int, 
        alpha: int = 3, 
        static_feature: Optional[Tensor] = None
    ):
        super(_GraphLearning, self).__init__()

        self.num_nodes = num_nodes
        self.k = k
        self.alpha = alpha
        self.static_feature = static_feature

        if self.static_feature is not None:
            static_dim = self.static_feature.shape[1]
            self.linear1 = nn.Linear(static_dim, embedding_dim)
            self.linear2 = nn.Linear(static_dim, embedding_dim)
        else:
            self.embedding1 = nn.Embedding(self.num_nodes, embedding_dim)
            self.embedding2 = nn.Embedding(self.num_nodes, embedding_dim)
            self.linear1 = nn.Linear(embedding_dim, embedding_dim)
            self.linear2 = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(self, idx: Tensor) -> Tensor:
        """
        Foward pass.

        Parameters:
            idx: input indices, a permutation of the number of nodes
        
        Return:
            A: adjacency matrix constructed by graph learning layer
        
        Shape:
            A: (N, N)
        """

        if self.static_feature is None:
            nodevec1 = self.embedding1(idx)
            nodevec2 = self.embedding2(idx)
        else:
            nodevec1 = self.static_feature[idx,:]
            nodevec2 = nodevec1
        
        nodevec1 = torch.tanh(self.alpha*self.linear1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.linear2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
        A = F.relu(torch.tanh(self.alpha * a))

        # For each node, select its top-k closest nodes as its neighbors
        mask = torch.zeros(idx.size(0), idx.size(0)).to(A.device)
        top_values, top_indices = (A + torch.rand_like(A)*0.01).topk(self.k, 1)
        mask.scatter_(1, top_indices, top_values.fill_(1))

        A = A * mask

        return A
    
class _LayerNormalization(nn.Module):
    """
    layer normalization layer.

    Parameters:
        normalized_shape: input shape from an expected input of size
        eps: value added to the denominator for numerical stability
        elementwise_affine: whether to conduct elementwise affine transformation
    """

    __constants__ = ["normalized_shape", "weight", "bias", "eps", "elementwise_affine"]

    def __init__(
        self, 
        normalized_shape: Union[int, Tuple], 
        eps: float = 1e-5, 
        elementwise_affine: bool = True
    ):
        super(_LayerNormalization, self).__init__()

        self.normalized_shape = tuple(normalized_shape)

        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, x: Tensor, idx: Tensor) -> Tensor:
        """
        Forward pass.

        Parameters:
            x: input features
            idx: input indices
        
        Shape:
            x: (B, C, N, T)
        """
        if self.elementwise_affine:
            return F.layer_norm(x, tuple(x.shape[1:]), self.weight[:,idx,:], self.bias[:,idx,:], self.eps)
        else:
            return F.layer_norm(x, tuple(x.shape[1:]), self.weight, self.bias, self.eps)