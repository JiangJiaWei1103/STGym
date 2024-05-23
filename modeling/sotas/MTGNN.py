"""
Baseline method, MTGNN [KDD, 2020].
Author: JiaWei Jiang

Reference:
* Paper: https://arxiv.org/abs/2005.11650
* Code: https://github.com/nnzhan/MTGNN
"""
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from modeling.module.tconv import TConvBaseModule
from modeling.sotas.gs_learner import MTGNNGSLearner
from modeling.sotas.layers import MTGNNLayer
from modeling.sotas.sub_layers import Linear2d


class MTGNN(TConvBaseModule):
    """MTGNN framework.

    Args:
        in_dim: input feature dimension
        skip_dim: output dimension of skip connection
        end_dim: hidden dimension of output layer
        in_len: input sequence length
        out_len: output sequence length
        n_series: number of series
        n_layers: number of GWNet layers
        tcn_in_dim: input dimension of GatedTCN
        gcn_in_dim: input dimension of GCN2d
        kernel_size: kernel size
        dilation_exponential: dilation exponential base
        n_adjs: number of transition matrices
        gcn_depth: depth of graph convolution
        gcn_dropout: dropout ratio of graph convolution
        bn: if True, apply batch normalization to output node embedding
            of graph convolution
    """

    def __init__(
        self,
        in_dim: int,
        skip_dim: int,
        end_dim: int,
        in_len: int,
        out_len: int,
        gsl_params: Dict[str, Any],
        st_params: Dict[str, Any],
    ) -> None:
        self.name = self.__class__.__name__
        super(MTGNN, self).__init__()

        # Network parameters
        # Graph learning layer
        self.gsl_params = gsl_params
        n_series = gsl_params["n_series"]
        node_emb_dim = gsl_params["node_emb_dim"]
        alpha = gsl_params["alpha"]
        k = gsl_params["k"]
        # Spatio-temporal pattern extractor
        self.st_params = st_params
        n_layers = st_params["n_layers"]
        tcn_in_dim = st_params["tcn_in_dim"]
        gcn_in_dim = st_params["gcn_in_dim"]
        kernel_size = st_params["kernel_size"]
        dilation_exponential = st_params["dilation_exponential"]
        n_adjs = st_params["n_adjs"]
        gcn_depth = st_params["gcn_depth"]
        beta = st_params["beta"]
        dropout = st_params["dropout"]
        ln_affine = st_params["ln_affine"]
        out_dim = out_len
        # Receptive field and sequence length
        self._set_receptive_field(n_layers, dilation_exponential, kernel_size[-1])
        if in_len < self.receptive_field:
            in_len = self.receptive_field

        # Model blocks
        # Input linear layer
        self.in_lin = Linear2d(in_dim, tcn_in_dim)
        # Self-adaptive adjacency matrix
        self.gs_learner = MTGNNGSLearner(n_series, node_emb_dim, alpha, k)
        self.node_idx = torch.arange(n_series)
        # Stacked spatio-temporal layers
        self.in_skip = nn.Sequential(
            nn.Dropout(dropout), nn.Conv2d(in_channels=tcn_in_dim, out_channels=skip_dim, kernel_size=(1, in_len))
        )
        self.mtgnn_layers = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        for layer in range(n_layers):
            self.mtgnn_layers.append(
                MTGNNLayer(
                    n_layers=layer + 1,
                    n_series=n_series,
                    in_len=in_len,
                    in_dim=tcn_in_dim,
                    h_dim=gcn_in_dim,
                    kernel_size=kernel_size,
                    dilation_exponential=dilation_exponential,
                    tcn_dropout=dropout,
                    n_adjs=n_adjs,
                    gcn_depth=gcn_depth,
                    beta=beta,
                    ln_affine=ln_affine,
                )
            )

            layer_out_len = self.mtgnn_layers[layer].out_len
            self.skip_convs.append(
                nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Conv2d(in_channels=gcn_in_dim, out_channels=skip_dim, kernel_size=(1, layer_out_len)),
                )
            )
        self.out_skip = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(
                in_channels=gcn_in_dim, out_channels=skip_dim, kernel_size=(1, in_len - self.receptive_field + 1)
            ),
        )
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
        """
        # Input linear layer
        x = x.permute(0, 3, 2, 1)  # (B, C, N, P)
        x = self._pad_seq_to_rf(x)
        x = self.in_lin(x)  # (B, tcn_in_dm, N, P)

        # Self-adaptive adjacency matrix
        if self.node_idx.device != x.device:
            self.node_idx = self.node_idx.to(x.device)
        A_adp = self.gs_learner(self.node_idx).to(x.device)

        # Stacked spatio-temporal layers
        x_skip = self.in_skip(x)
        for layer, mtgnn_layer in enumerate(self.mtgnn_layers):
            # Capture inflow and outflow patterns
            h_tcn, x = mtgnn_layer(x, [A_adp, A_adp.T])

            # Skip-connect to the output module
            x_skip = x_skip + self.skip_convs[layer](h_tcn)  # (B, skip_dim, N, 1)

        # Output layer
        x_skip = x_skip + self.out_skip(x)
        output = self.output(x_skip).squeeze(dim=-1)  # (B, Q, N)

        return output, None, None
