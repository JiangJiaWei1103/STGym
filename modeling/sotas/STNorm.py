"""
Baseline method, ST-Norm [KDD, 2021].
Author: ChunWei Shen

Reference:
* Paper: https://dl.acm.org/doi/10.1145/3447548.3467330
* Code: https://github.com/JLDeng/ST-Norm
"""
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from modeling.module.layers import GWNetLayer
from modeling.module.common_layers import Linear2d, SNorm, TNorm


class STNorm(nn.Module):
    """STNorm framework.

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
        bn: if True, apply batch normalization to output node embedding
            of graph convolution
    """

    def __init__(
        self, in_dim: int, skip_dim: int, end_dim: int, out_len: int, n_series: int, st_params: Dict[str, Any]
    ) -> None:
        self.name = self.__class__.__name__
        super(STNorm, self).__init__()

        # Network parameters
        self.st_params = st_params
        # Spatio-temporal pattern extractor
        n_layers = st_params["n_layers"]
        tcn_in_dim = st_params["tcn_in_dim"]
        gcn_in_dim = st_params["gcn_in_dim"]
        kernel_size = st_params["kernel_size"]
        dilation_factor = st_params["dilation_factor"]
        self.snorm = st_params["snorm"]
        self.tnorm = st_params["tnorm"]
        bn = st_params["bn"]
        if isinstance(dilation_factor, list):
            assert len(dilation_factor) == n_layers, "Layer-wise dilation factors aren't aligned."
        out_dim = out_len

        # Model blocks
        # Input linear layer
        self.in_lin = Linear2d(in_dim, tcn_in_dim)

        # Stacked spatio-temporal layers
        self._receptive_field = 1
        self.snorm_layers = nn.ModuleList()
        self.tnorm_layers = nn.ModuleList()
        self.wavenet_layers = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        for layer in range(n_layers):
            if isinstance(dilation_factor, list):
                d = dilation_factor[layer]
            else:
                d = pow(dilation_factor, layer)
            self._receptive_field += d * (kernel_size - 1)

            # ST-Norm
            if self.snorm:
                self.snorm_layers.append(SNorm(in_dim=tcn_in_dim))
            if self.tnorm:
                self.tnorm_layers.append(TNorm(in_dim=tcn_in_dim, n_series=n_series))
            in_dim = (1 + int(self.tnorm) + int(self.snorm)) * tcn_in_dim

            self.wavenet_layers.append(
                GWNetLayer(
                    in_dim=in_dim,
                    h_dim=gcn_in_dim,
                    kernel_size=kernel_size,
                    dilation_factor=d,
                    gcn = False,
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

        # Stacked spatio-temporal layers
        x_skips = []
        for layer, wavenet_layer in enumerate(self.wavenet_layers):
            x_list = [x]
            # ST-Norm
            if self.snorm:
                x_list.append(self.snorm_layers[layer](x))
            if self.tnorm:
                x_list.append(self.tnorm_layers[layer](x))    
            x = torch.cat(x_list, dim=1)

            h_tcn, x = wavenet_layer(x)
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