"""
STSGCN framework.

Reference: 
https://github.com/Davidham3/STSGCN
https://github.com/j1o2h3n/STSGCN

Author: ChunWei Shen
"""
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import init
import torch.nn.functional as F
import numpy as np

class STSGCN(nn.Module):
    """
    STSGCN.

    Parameters:
        st_params: hyperparameters of Spatial/Temporal pattern extractor
        emb_params: hyperparameters of Spatial-Temporal embedding
        out_dim: output dimension
        device: device
        priori_gs: predefined adjacency matrix
    """

    def __init__(
        self,
        st_params: Dict[str, Any],
        emb_params: Dict[str, Any],
        device: str,
        out_dim: int,
        priori_gs: Optional[List[Tensor]] = None
    ):
        super(STSGCN, self).__init__()

        # Network parameters
        self.st_params = st_params
        self.emb_params = emb_params
        self.out_dim = out_dim
        self.A = self._construct_adj(priori_gs[0]).to(device)

        # hyperparameters of Spatial/Temporal pattern extractor
        n_series = self.st_params['n_series']
        t_window = self.st_params['t_window']
        in_channels = self.st_params['in_channels']
        hid_dim  = self.st_params['hid_dim']
        act_func = self.st_params['act_func']
        self.filters = self.st_params['filters']
        # hyperparameters of Spatial-Temporal embedding
        temporal_bool = self.emb_params['temporal_bool']
        spatial_bool = self.emb_params['spatial_bool']
        
        self.mask = nn.Parameter(torch.rand(3*n_series, 3*n_series).to(device), requires_grad=True)

        self.stsgcl = nn.ModuleList()
        for filter in self.filters:
            self.stsgcl.append(_STSGCL(
                n_series,
                t_window,
                filter,
                hid_dim,
                act_func,
                temporal_bool,
                spatial_bool))
            t_window -= 2
            hid_dim = filter[-1]

        self.output_layer = nn.ModuleList()
        for _ in range(self.out_dim):
            self.output_layer.append(_output_layer(hid_dim, t_window))

        self.input_layer= torch.nn.Conv2d(
            in_channels,
            hid_dim,
            kernel_size=(1, 1),
            padding=(0, 0),
            stride=(1, 1),
            bias=True)

    def forward(
        self,
        input: Tensor,
        **kwargs: Any,
    ) -> Tuple[Tensor, None, None]:
        """
        Forward pass.

        Parameters:
            input: input features

        Return:
            output: prediction

        Shape:
            input: (B, T, N, C)
            output: (B, out_dim, N)
        """
        input = input.permute(0, 3, 2, 1)   # (B, C, N, T)
        data = self.input_layer(input)      # (B, hid_dim, N, T)
        data = torch.relu(data)
        data = data.permute(0, 3, 2, 1)     # (B, T, N, hid_dim)

        adj = self.mask * self.A

        for i in range(len(self.filters)):
            data = self.stsgcl[i](data, adj)
        # (B, T-2*layer_num, N, C')
        
        need_concat = []
        for i in range(self.out_dim):
            output = self.output_layer[i](data)      # (B, 1, N)
            need_concat.append(output.squeeze(1))

        outputs = torch.stack(need_concat, dim = 1)  # (B, 12, N)

        return outputs, None, None
    
    def _construct_adj(
        self, 
        A: Tensor, 
        steps: int = 3
    ) -> Tensor:
        """
        Construct adjacency matrix.

        Parameters:
            A: adjacency matrix
            steps: number of step

        Shape:
            A: (N, N)
            adj: (step*N, step*N)
        """
        N = len(A)
        adj = np.zeros([N * steps] * 2)

        for i in range(steps):
            adj[i * N: (i + 1) * N, i * N: (i + 1) * N] = A

        for i in range(N):
            for k in range(steps - 1):
                adj[k * N + i, (k + 1) * N + i] = 1
                adj[(k + 1) * N + i, k * N + i] = 1

        for i in range(len(adj)):
            adj[i, i] = 1

        adj = torch.from_numpy(adj.astype(np.float32))

        return adj

class _output_layer(nn.Module):
    """
    Output layer.

    Parameters:
        hid_dim: hidden dimension
        t_window: lookback time window
    """
    def __init__(
        self,
        hid_dim: int,
        t_window: int
    ):
        super(_output_layer,self).__init__()
        
        self.fully_1 = nn.Conv2d(t_window * hid_dim, 128, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)
        self.fully_2 = nn.Conv2d(128, 1, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(
        self, 
        x: Tensor
    ) -> Tensor:
        """
        Forward pass.

        Parameters:
            x: input features

        Shape:
            x: (B, T, N, C)
            output: (B, 1 ,N)
        """
        _, T, N, C = x.size()

        x = x.permute(0, 2, 1, 3)           # (B, T, N, C)->(B, N, T, C)
        x = x.reshape([-1, N, T * C, 1])    # (B, N, T, C)->(B, N, T*C, 1)
        x = x.permute(0, 2, 1, 3)           # (B, N, T*C, 1)->(B, T*C, N, 1)

        x = self.fully_1(x)                 # (B, 128, N, 1)
        x = torch.relu(x)
        x = self.fully_2(x)                 # (B, 1, N, 1)

        output = x.squeeze(dim = 3)         # (B, 1, N)

        return output

class _STSGCL(nn.Module):
    """
    Spatial-Temporal Synchronous Graph Convolutional Layer.

    Parameters:
        n_series: number of nodes
        t_window: lookback time window
        filters: list of gcn filters(output channels)
        c_in: input channels
        act_func: activation function
        temporal_bool: whether to add temporal embedding
        spatial_bool: whether to add spatial embedding
    """
    def __init__(
        self,
        n_series: int,
        t_window: int,
        filters: List[int],
        c_in: int,
        act_func: str,
        temporal_bool: bool,
        spatial_bool: bool
    ):
        super(_STSGCL, self).__init__()

        self.c_in = c_in
        self.T = t_window
        self.n_series = n_series
        
        self.stsgcm = nn.ModuleList()

        # Spatial-Temporal Synchronous Graph Convolutional Module
        for _ in range(self.T - 2):
            self.stsgcm.append(_STSGCM(n_series, filters, c_in, act_func))

        # Spatial Temporal embedding
        self.emb = _Spatial_Temporal_embedding(t_window, n_series, c_in, temporal_bool, spatial_bool)
        
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

        Shape:
            x: (B, T, N, C)
            A: (3N, 3N)
            outputs: (B, T-2, N, C')
        """
        need_concat = []

        # Spatial Temporal embedding
        x = self.emb(x)
        data = x
        for i in range(self.T - 2):
            t = data[:,i:i+3,:,:]                               # (B, 3, N, C)
            t = t.reshape([-1, 3 * self.n_series, self.c_in])   # (B, 3N, C)
            t = t.permute(1, 0, 2)                              # (3N, B, C)
            t = self.stsgcm[i](t, A)                            # (N, B, C')
            t = t.permute(1, 0, 2)                              # (B, N, C')
            need_concat.append(t)

        outputs = torch.stack(need_concat, dim = 1)             # (B, T-2, N, C')

        return outputs

class _STSGCM(nn.Module):
    """
    Spatial-Temporal Synchronous Graph Convolutional Module.

    Parameters:
        n_series: number of nodes
        filters: list of gcn filters(output channels)
        c_in: input channels
        act_func: activation function
    """
    def __init__(
        self,
        n_series: int, 
        filters: List[int],
        c_in: int,
        act_func: str
    ):
        super(_STSGCM,self).__init__()

        self.gcn = nn.ModuleList()

        for i in range(len(filters)):
            self.gcn.append(_GCN(c_in, filters[i], act_func))
            c_in = filters[i]
        
        self.num_nodes = n_series
        self.num_gcn = len(filters)

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

        Shape:
            x: (3N, B, C)
            A: (3N, 3N)
            output: (N, B, C')
        """
        need_concat = []

        for i in range(self.num_gcn):
            x = self.gcn[i](x, A)       # (3N, B, C')
            need_concat.append(x)
        
        # Aggregating operation and Cropping operation
        need_concat = [i[(self.num_nodes):(2*self.num_nodes),:,:].unsqueeze(0) for i in need_concat] # (1, N, B, C')
        outputs = torch.cat(need_concat, dim = 0)        # (3, N, B, C')
        outputs = torch.max(outputs, dim = 0).values    # (N, B, C')

        return outputs

class _GCN(nn.Module):
    """
    Graph Convolution.

    Parameters:
        c_in: input channels
        c_out: outpupt channels
        act_func: activation function
    """
    def __init__(
        self,
        c_in: int,
        c_out: int,
        act_func: str
    ):
        super(_GCN,self).__init__()

        if act_func == "GLU":
            self.mlp = _Linear(c_in, 2 * c_out)
        elif act_func == "Relu":
            self.mlp = _Linear(c_in, c_out)

        self.act_func = act_func
        self.c_out = c_out

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

        Shape:
            x: (3N, B, C)
            A: (3N, 3N)
            output: (3N, B, C')
        """

        x = x.unsqueeze(-1)                         # (3N, B, C, 1)
        x = x.permute(1, 2, 0, 3)                   # (3N, B, C, 1) -> (B, C, 3N, 1)
        x = torch.einsum("vw,ncwl->ncvl", (A, x))   # (B, C, 3N, 1)
        x = self.mlp(x)                             # (B, 2C', 3N, 1) or (B, C', 3N, 1)

        if self.act_func == "GLU":
            lhs, rhs = torch.split(x, self.c_out, dim = 1)  # (B, C', 3N, 1), (B, C', 3N, 1)
            output = lhs * torch.sigmoid(rhs)               # (B, C', 3N, 1)
            output = output.squeeze(3).permute(2, 0, 1)     # (3N, B, C')
        elif self.act_func == "Relu":
            output = F.relu(x.squeeze(3).permute(2, 0, 1))

        return output

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

class _Spatial_Temporal_embedding(nn.Module):
    """
    Spatial Temporal embedding.

    Parameters:
        t_window: lookback time window
        n_series: number of nodes
        embedding_size: embedding size
        temporal: whether to add temporal embedding
        spatial: whether to add spatial embedding
    """
    def __init__(
        self,
        t_window: int,
        n_series: int,
        embedding_size: int,
        temporal: bool = True,
        spatial: bool = True
    ):
        super(_Spatial_Temporal_embedding, self).__init__()

        self.temporal_emb = None
        self.spatial_emb = None

        # temporal embedding
        if temporal:
            self.temporal_emb = init.xavier_normal_(
                torch.empty(1, t_window, 1, embedding_size),
                gain = 0.0003)
        # spatial embedding
        if spatial:
            self.spatial_emb = init.xavier_normal_(
                torch.empty(1, 1, n_series, embedding_size),
                gain = 0.0003)
            
    def forward(
        self,
        x: Tensor
    ) -> Tensor:
        """
        Forward pass.

        Parameters:
            x: input features
        
        Shape:
            x: (B, T, N, C)
        """

        # temporal embedding
        if self.temporal_emb is not None:
            x = x + self.temporal_emb.to(x.device)
        # spatial embedding
        if self.spatial_emb is not None:
            x = x + self.spatial_emb.to(x.device)
            
        return x