"""
Sub-layers.
Author: JiaWei Jiang
"""
from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor


class DiffusionConvLayer(nn.Module):
    """Diffusion convolutional layer."""

    def __init__(
        self, in_dim: int, h_dim: int, n_adjs: int = 2, max_diffusion_step: int = 2, act: Optional[str] = None
    ) -> None:
        super(DiffusionConvLayer, self).__init__()

        # Network parameters
        self.in_dim = in_dim
        self.max_diffusion_step = max_diffusion_step
        self.n_node_embs = 1 + n_adjs * max_diffusion_step  # Number of node embeddings (i.e., feature matrices)

        # Model blocks
        self.conv_filter = nn.Linear(in_dim * self.n_node_embs, h_dim)
        if act is not None:
            if act == "relu":
                self.act = nn.ReLU()
            elif act == "sigmoid":
                self.act = nn.Sigmoid()
        else:
            self.act = None

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_normal_(self.conv_filter.weight.data)
        nn.init.constant_(self.conv_filter.bias.data, val=0.0)

    def forward(self, x: Tensor, As: List[Tensor]) -> Tensor:
        """Forward pass.

        Parameters:
            x: input node features
            As: list of adjacency matrices

        Return:
            h: diffusion convolution output

        Shape:
            x: (B, N, C)
            As: each A with shape (2, |E|), where |E| denotes the
                number edges
            x_conv: (B, N, C')
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
                    x_conv = 2 * torch.mm(A, x_conv) - x
                    x = x_conv
                x_convs = torch.cat([x_convs, x_conv.unsqueeze(0)], dim=0)

        x = x_convs.reshape(self.n_node_embs, n_series, self.in_dim, batch_size)
        x = x.permute(3, 1, 2, 0)  # (B, N, C, n_node_embs)
        x = x.reshape(batch_size * n_series, -1)
        h = self.conv_filter(x).reshape(batch_size, n_series, -1)  # (B, N, h_dim)

        return h
