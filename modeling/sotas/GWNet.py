"""
Baseline method, GWNet [IJCAI, 2019].
Author: JiaWei Jiang, ChunWei Shen

Reference:
* Paper: https://arxiv.org/abs/1906.00121
* Code: https://github.com/nnzhan/Graph-WaveNet
"""
from omegaconf import ListConfig
from typing import Any, Dict, List, Tuple

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from modeling.module.gs_learner import GWNetGSLearner
from modeling.module.layers import GWNetLayer
from modeling.module.common_layers import Linear2d


class GWNet(nn.Module):
    """GWNet framework.

    Args:
        in_dim: input feature dimension
        skip_dim: output dimension of skip connection
        end_dim: hidden dimension of output layer
        out_len: output sequence length
        n_series: number of series
        n_layers: number of GWNet layers
        tcn_in_dim: input dimension of GatedTCN
        gcn_in_dim: input dimension of GCN2d
        kernel_size: kernel size
        dilation_factor: layer-wise dilation factor or exponential base
        n_adjs: number of transition matrices
        gcn_depth: depth of graph convolution
        gcn_dropout: dropout ratio of graph convolution
        bn: if True, apply batch normalization to output node embedding
            of graph convolution
    """

    def __init__(
        self, in_dim: int, skip_dim: int, end_dim: int, out_len: int, n_series: int, st_params: Dict[str, Any]
    ) -> None:
        self.name = self.__class__.__name__
        super(GWNet, self).__init__()

        # Network parameters
        self.st_params = st_params
        # Spatio-temporal pattern extractor
        n_layers = st_params["n_layers"]
        tcn_in_dim = st_params["tcn_in_dim"]
        gcn_in_dim = st_params["gcn_in_dim"]
        kernel_size = st_params["kernel_size"]
        dilation_factor = st_params["dilation_factor"]
        n_adjs = st_params["n_adjs"]
        gcn_depth = st_params["gcn_depth"]
        gcn_dropout = st_params["gcn_dropout"]
        bn = st_params["bn"]
        if isinstance(dilation_factor, list):
            assert len(dilation_factor) == n_layers, "Layer-wise dilation factors aren't aligned."
        out_dim = out_len

        # Model blocks
        # Input linear layer
        self.in_lin = Linear2d(in_dim, tcn_in_dim)
        # Self-adaptive adjacency matrix
        self.gs_learner = GWNetGSLearner(n_series)
        # Stacked spatio-temporal layers
        self._receptive_field = 1
        self.gwnet_layers = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        for layer in range(n_layers):
            if isinstance(dilation_factor, ListConfig):
                d = dilation_factor[layer]
            else:
                d = pow(dilation_factor, layer)
            self._receptive_field += d * (kernel_size - 1)

            self.gwnet_layers.append(
                GWNetLayer(
                    in_dim=tcn_in_dim,
                    h_dim=gcn_in_dim,
                    kernel_size=kernel_size,
                    dilation_factor=d,
                    n_adjs=n_adjs,
                    gcn_depth=gcn_depth,
                    gcn_dropout=gcn_dropout,
                    bn=bn,
                )
            )
            self.skip_convs.append(Linear2d(gcn_in_dim, skip_dim))
        # Output layer
        self.output = nn.Sequential(nn.ReLU(), Linear2d(skip_dim, end_dim), nn.ReLU(), Linear2d(end_dim, out_dim))

    def forward(self, x: Tensor, As: List[Tensor], **kwargs: Any) -> Tuple[Tensor, None, None]:
        """Forward pass.

        Args:
            x: input sequence
            As: list of adjacency matrices

        Returns:
            output: prediction

        Shape:
            x: (B, P, N, C)
            As: each A with shape (N, N)
            output: (B, Q, N)
        """
        # Input linear layer
        x = x.permute(0, 3, 2, 1)  # (B, C, N, P)
        x = self._pad_seq_to_receptive(x)
        x = self.in_lin(x)

        # Self-adaptive adjacency matrix
        A_adp = self.gs_learner().to(x.device)
        As_aug = As + [A_adp]

        # Stacked spatio-temporal layers
        x_skips = []
        for layer, gwnet_layer in enumerate(self.gwnet_layers):
            h_tcn, x = gwnet_layer(x, As_aug)
            x_skip = self.skip_convs[layer](h_tcn)  # (B, skip_dim, N, L')
            x_skips.append(x_skip)

        # Output layer
        assert x_skip.shape[-1] == 1, "Temporal dimension must be equal to 1."
        x = x_skip  # Last skip component
        for x_skip in x_skips[:-1]:
            x = x + x_skip[..., -1].unsqueeze(dim=-1)  # (B, skip_dim, N, 1)
        output = self.output(x).squeeze(dim=-1)  # (B, Q, N)

        return output, None, None

    def _pad_seq_to_receptive(self, x: Tensor) -> Tensor:
        """Pad sequence to the receptive field."""
        in_len = x.shape[-1]
        if in_len < self._receptive_field:
            x = F.pad(x, (self._receptive_field - in_len, 0))

        return x
