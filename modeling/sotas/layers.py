"""
Common sptio-temporal layers.
Author: JiaWei Jiang
"""
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .sub_layers import DiffusionConvLayer


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
