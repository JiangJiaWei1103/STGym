"""
Common sptio-temporal layers.
Author: JiaWei Jiang
"""
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .sub_layers import DiffusionConvLayer, GatedTCN, GCN2d


class DCGRU(nn.Module):
    """Diffusion convolutional gated recurrent unit.

    Parameters:
        in_dim: input feature dimension
        h_dim: hidden state dimension
        n_adjs: number of adjacency matrices
            *Note: Bidirectional transition matrices are used in the
                original proposal.
        max_diffusion_step: maximum diffusion step
        act: activation function
    """

    def __init__(
        self, in_dim: int, h_dim: int, n_adjs: int = 2, max_diffusion_step: int = 2, act: Optional[str] = None
    ) -> None:
        super(DCGRU, self).__init__()

        # Network parameters
        self.h_dim = h_dim

        # Model blocks
        cat_dim = in_dim + h_dim
        self.gate = DiffusionConvLayer(
            in_dim=cat_dim, h_dim=h_dim * 2, n_adjs=n_adjs, max_diffusion_step=max_diffusion_step, act=act
        )
        self.candidate_act = DiffusionConvLayer(
            in_dim=cat_dim, h_dim=h_dim, n_adjs=n_adjs, max_diffusion_step=max_diffusion_step, act=act
        )

    def forward(self, x: Tensor, As: List[Tensor], h_0: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """Forward pass.

        Parameters:
            x: input sequence
            As: list of adjacency matrices
            h_0: initial hidden state

        Return:
            output: hidden state for each lookback time step
            h_n: last hidden state

        Shape:
            x: (B, L, N, C), where L denotes the input sequence length
            As: each A with shape (2, |E|), where |E| denotes the
                number edges
            h_0: (B, N, h_dim)
            output: (B, L, N, h_dim)
            h_n: (B, N, h_dim)
        """
        in_len = x.shape[1]

        output = []
        for t in range(in_len):
            x_t = x[:, t, ...]  # (B, N, C)
            if t == 0:
                h_t = None
                h_prev = self._init_hidden_state(x) if h_0 is None else h_0
            else:
                h_prev = h_t

            gate = F.sigmoid(self.gate(torch.cat([h_prev, x_t], dim=-1), As))
            r_t, u_t = torch.split(gate, self.h_dim, dim=-1)  # (B, N, h_dim)
            c_t = F.tanh(self.candidate_act(torch.cat([r_t * h_prev, x_t], dim=-1), As))
            h_t = u_t * h_prev + (1 - u_t) * c_t  # (B, N, h_dim)

            output.append(h_t.unsqueeze(dim=1))
        output = torch.cat(output, dim=1)  # (B, L, N, h_dim)
        h_n = h_t

        return output, h_n

    def _init_hidden_state(self, x: Tensor) -> Tensor:
        """Initialize the initial hidden state."""
        batch_size, _, n_series = x.shape[:-1]
        h_0 = torch.zeros(batch_size, n_series, self.h_dim, device=x.device)

        return h_0


class GWNetLayer(nn.Module):
    """Spatio-temporal layer of GWNet.

    One layer is constructed by a graph convolution layer and a gated
    temporal convolution layer.

    Parameters:
        in_dim: input feature dimension
        h_dim: hidden dimension
        kernel_size: kernel size
        dilation_factor: dilation factor
        n_adjs: number of adjacency matrices
            *Note: Bidirectional transition matrices are used in the
                original proposal.
        gcn_depth: depth of graph convolution
        gcn_dropout: droupout ratio in graph convolution layer
        bn: if True, apply batch normalization to output node embedding
    """

    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        kernel_size: int,
        dilation_factor: int,
        n_adjs: int = 3,
        gcn_depth: int = 2,
        gcn_dropout: float = 0.3,
        bn: bool = True,
    ) -> None:
        super(GWNetLayer, self).__init__()

        # Model blocks
        # Gated temporal convolution layer
        self.tcn = GatedTCN(in_dim=in_dim, h_dim=h_dim, kernel_size=kernel_size, dilation_factor=dilation_factor)
        # Graph convolution layer
        self.gcn = GCN2d(in_dim=h_dim, h_dim=in_dim, n_adjs=n_adjs, depth=gcn_depth, dropout=gcn_dropout)
        if bn:
            self.bn = nn.BatchNorm2d(in_dim)
        else:
            self.bn = None

    def forward(self, x: Tensor, As: List[Tensor]) -> Tensor:
        """Forward pass.

        Parameters:
            x: input sequence
            As: list of adjacency matrices

        Return:
            h_tcn: intermediate node embedding output by GatedTCN
            h: output node embedding

        Shape:
            x: (B, C, N, L), where L denotes the input sequence length
            As: each A with shape (N, N)
            h_tcn: (B, h_dim, N, L')
            h: (B, C, N, L')
        """
        x_resid = x

        # Gated temporal convolution layer
        h_tcn = self.tcn(x)

        # Graph convolution layer
        h = self.gcn(h_tcn, As)

        out_len = h.shape[-1]
        h = h + x_resid[..., -out_len:]
        if self.bn is not None:
            h = self.bn(h)

        return h_tcn, h
