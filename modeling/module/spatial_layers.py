"""
Spatial-layers.
Author: JiaWei Jiang, ChunWei Shen
"""
from typing import List, Optional, Type, Union

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .common_layers import Linear2d, Align

class DiffusionConvLayer(nn.Module):
    """Diffusion convolutional layer."""

    def __init__(
        self, in_dim: int, h_dim: int, n_adjs: int = 2, max_diffusion_step: int = 2
    ) -> None:
        super(DiffusionConvLayer, self).__init__()

        # Network parameters
        self.in_dim = in_dim
        self.max_diffusion_step = max_diffusion_step
        self.n_node_embs = 1 + n_adjs * max_diffusion_step  # Number of node embeddings (i.e., feature matrices)

        # Model blocks
        self.conv_filter = nn.Linear(in_dim * self.n_node_embs, h_dim)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_normal_(self.conv_filter.weight.data)
        nn.init.constant_(self.conv_filter.bias.data, val=0.0)

    def forward(self, x: Tensor, As: List[Tensor]) -> Tensor:
        """Forward pass.

        Args:
            x: input node features
            As: list of adjacency matrices

        Returns:
            h: diffusion convolution output

        Shape:
            x: (B, N, C)
            As: each A with shape (N, N)
            h: (B, N, h_dim)
        """
        batch_size, n_series, _ = x.shape

        x = x.permute(1, 2, 0)  # (N, C, B)
        x = x.reshape(n_series, -1)  # (N, C * B)
        x_convs = x.unsqueeze(0)  # (1, N, C * B)
        for A in As:
            for k in range(1, self.max_diffusion_step + 1):
                if k == 1:
                    x_conv = torch.mm(A, x)
                else:
                    tmp = x_conv
                    x_conv = 2 * torch.mm(A, x_conv) - x
                    x = tmp
                x_convs = torch.cat([x_convs, x_conv.unsqueeze(0)], dim=0)

        x = x_convs.reshape(self.n_node_embs, n_series, self.in_dim, batch_size)
        x = x.permute(3, 1, 2, 0)  # (B, N, C, n_node_embs)
        x = x.reshape(batch_size * n_series, -1)
        h = self.conv_filter(x).reshape(batch_size, n_series, -1)  # (B, N, h_dim)

        return h


class ChebGraphConv(nn.Module):
    """Chebyshev graph convolution."""

    def __init__(self, in_dim: int, h_dim: int, cheb_k: int, bias: bool = True) -> None:
        super(ChebGraphConv, self).__init__()

        # Network parameters
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.cheb_k = cheb_k

        # Model blocks
        self.align = Align(in_dim, h_dim)
        self.gconv = GCN2d(flow="tgt_to_src")
        self.weight = nn.Parameter(torch.FloatTensor(cheb_k, h_dim, h_dim))
        self.bias = nn.Parameter(torch.FloatTensor(h_dim)) if bias else None

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a = math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: Tensor, As: List[Tensor]) -> Tensor:
        """Forward pass.

        Args:
            x: input node features
            As: list of adjacency matrices

        Returns:
            h: chebyshev graph convolution output
        
        Shape:
            x: (B, C, N, L)
            h: (B, C', N, L)
        """
        x_resid = self.align(x)
        x = x_resid.permute(0, 3, 2, 1)    # (B, L, N, C)

        x_convs = [x]
        x_conv = x
        for i in range(1, self.cheb_k):
            if i == 1:
                x_conv = self.gconv(x_conv, As[0])
            else:
                tmp = x_conv
                x_conv = self.gconv(x_conv, As[0]) - x
                x = tmp
            x_convs.append(x_conv)
        
        x = torch.stack(x_convs, dim=2)
        
        h = torch.einsum('btkhi,kij->bthj', x, self.weight)     # (B, L, N, C')
        if self.bias is not None:
            h = torch.add(h, self.bias).permute(0, 3, 2, 1)     # (B, C', N, L)

        h = h + x_resid
        
        return h


class InfoPropLayer(nn.Module):
    """Information propagation layer used in GWNet and MTGNN.

    Args:
        flow: flow direction of the message passing, the choices are
            {"src_to_tgt", "tgt_to_src"}
        mix_prop: if True, mix-hop propagation in MTGNN is used
    """
    
    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        n_adjs: int,
        depth: int,
        flow: str = "src_to_tgt",
        normalize: Optional[str] = None,
        mix_prop: bool = False,
        beta: Optional[float] = None,
        dropout: Optional[float] = None,
    ) -> None:
        super(InfoPropLayer, self).__init__()

        # Network parameters
        self.depth = depth
        self.mix_prop = mix_prop
        if mix_prop:
            assert beta is not None, "Please specify retraining ratio for preserving locality."
        self.beta = beta
        self.n_node_embs = 1 + n_adjs * depth  # Number of node embeddings (i.e., feature matrices)

        # Model blocks
        self.gconv = GCN2d(flow=flow, normalize=normalize)
        self.conv_filter = Linear2d(in_dim * self.n_node_embs, h_dim)
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x: Tensor, As: List[Tensor]) -> Tensor:
        """Forward pass.

        Args:
            x: input node features
            As: list of adjacency matrices

        Returns:
            h: graph convolution output

        Shape:
            x: (B, in_dim, N, L)
            As: each A with shape (N, N)
            h: (B, h_dim, N, L)
        """
        # Information propagation
        x_convs = [x]  # k == 0
        x_conv = None
        h_in = x if self.mix_prop else None
        for A in As:
            for k in range(1, self.depth + 1):
                x_conv = x if k == 1 else x_conv
                if self.mix_prop:
                    x_conv = self.beta * h_in + (1 - self.beta) * self.gconv(x_conv, A)
                else:
                    x_conv = self.gconv(x_conv, A)
                x_convs.append(x_conv)

        # Information selection
        h = torch.cat(x_convs, dim=1)
        h = self.conv_filter(h)
        if self.dropout is not None:
            h = self.dropout(h)

        return h


class STSGCM(nn.Module):
    """Spatial-Temporal Synchronous Graph Convolutional Module."""

    def __init__(self, in_dim: int, h_dim: int, gcn_depth: int, n_series: int, act: str) -> None:
        super(STSGCM,self).__init__()

        # Network parameters
        self.gcn_depth = gcn_depth
        self.n_series = n_series
        self.act = act

        # Model blocks
        self.gconv = GCN2d(flow="tgt_to_src")
        self.conv_filter = nn.ModuleList()
        for i in range(gcn_depth):
            in_dim = in_dim if i == 0 else h_dim
            if act == "glu":
                self.conv_filter.append(Linear2d(in_features=in_dim, out_features=2 * h_dim))
            elif act == "relu":
                self.conv_filter.append(Linear2d(in_features=in_dim, out_features=h_dim))

    def forward(self, x: Tensor, A: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: input sequence
            A: adjacency matrix

        Shape:
            x: (3N, B, C)
            A: (3N, 3N)
            output: (N, B, C')
        """
        x = x.permute(1, 2, 0)     # (B, C, 3N)
        x_convs = []

        for i in range(self.gcn_depth):
            x = x.unsqueeze(-1)             # (B, C, 3N, 1)
            x = self.gconv(x, A)            # (B, C, 3N, 1)
            x = self.conv_filter[i](x)      # (B, 2C', 3N, 1) or (B, C', 3N, 1)
            if self.act == "glu":
                lhs, rhs = torch.split(x, x.shape[1] // 2, dim=1)    # (B, C', 3N, 1), (B, C', 3N, 1)
                x = lhs * torch.sigmoid(rhs)                         # (B, C', 3N, 1)
                x = x.squeeze(dim=-1)                                # (B, C', 3N)
            elif self.act == "relu":
                x = F.relu(x.squeeze(dim=-1))                        # (B, C', 3N)
            x_convs.append(x.permute(2, 0, 1))
        
        # Aggregating operation and Cropping operation
        x_convs = [conv[(self.n_series):(2 * self.n_series), :, :].unsqueeze(0) for conv in x_convs] # (1, N, B, C')
        output = torch.cat(x_convs, dim=0)          # (gcn_depth, N, B, C')
        output = torch.max(output, dim=0).values    # (N, B, C')

        return output


class NAPLConvLayer(nn.Module):
    """ Node Adaptive Parameter Learning Graph Convolution Layer."""

    def __init__(self, in_dim: int, h_dim: int, emb_dim: int, cheb_k: int) -> None:
        super(NAPLConvLayer, self).__init__()

        # Network parameters
        self.cheb_k = cheb_k

        # Model blocks
        self.weights_pool = nn.Parameter(torch.FloatTensor(emb_dim, cheb_k, in_dim, h_dim))
        self.bias_pool = nn.Parameter(torch.FloatTensor(emb_dim, h_dim))

    def forward(self, x: Tensor, node_embs: Tensor, As: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: input node embeddings
            node_embs: node embeddings

        Returns:
            x_conv: output node embeddings

        Shape:
            x: (B, N, C)
            node_emb: (N, D)
            x_conv: (B, N, h_dim)
        """
        weights = torch.einsum('nd,dkio->nkio', (node_embs, self.weights_pool))   # (N, cheb_k, in_dim, out_dim)
        bias = torch.matmul(node_embs, self.bias_pool)    # (N, out_dim)

        x_conv = torch.einsum("knm,bmc->bknc", As, x).permute(0, 2, 1, 3)   # (B, N, cheb_k, in_dim)
        x_conv = torch.einsum('bnki,nkio->bno', x_conv, weights) + bias     # (B, N, out_dim)

        return x_conv


class GCN2d(nn.Module):
    """Graph convolution layer over 2D planes.

    GCN2d applies graph convolution over graph signal represented by
    2D node embedding planes.
    """

    def __init__(
        self,
        flow: str = "src_to_tgt",
        normalize: Optional[str] = None,
    ) -> None:
        super(GCN2d, self).__init__()

        self.flow = flow
        self.normalize = normalize
        if flow == "src_to_tgt":
            self._flow_eq = "bcvl,vw->bcwl"
        else:
            self._flow_eq = "bcwl,vw->bcvl"

    def forward(self, x: Tensor, A: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: input node embeddings
            A: adjacency matrix

        Returns:
            h: output node embeddings

        Shape:
            x: (B, C, N, L)
            h: (B, C, N, L), the shape is the same as the input
        """
        if self.normalize is not None:
            A = self._normalize(A)
        h = torch.einsum(self._flow_eq, (x, A))

        return h

    def _normalize(self, A: Tensor) -> Tensor:
        """Normalize adjacency matrix."""
        if self.flow == "src_to_tgt":
            # Do column normalization
            A = A.T

        # Add self-loop
        A = A + torch.eye(A.size(0)).to(A.device)

        # Normalize
        if self.normalize == "asym":
            D = torch.sum(A, dim=1).reshape(-1, 1)
            A = A / D

        if self.flow == "src_to_tgt":
            A = A.T

        return A