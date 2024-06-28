"""
Baseline method, AGCRN [NeurIPS, 2020].
Author: ChunWei Shen

Reference:
* Paper: https://arxiv.org/abs/2007.02842
* Code: https://github.com/LeiBAI/AGCRN
"""
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from modeling.module.layers import AGCGRU
from modeling.module.gs_learner import AGCRNGSLearner


class AGCRN(nn.Module):
    """AGCRN framework.

    Args:
        in_dim: input feature dimension
        out_dim: output dimension
        out_len: output sequence length
        n_layers: number of AGCRN layers
        h_dim: hidden dimension
        cheb_k: order of chebyshev polynomial expansion
        n_series: number of series
        emb_dim: dimension of node embedding
    """

    def __init__(
        self, in_dim: int, out_dim: int, out_len: int, st_params: Dict[str, Any], dagg_params: Dict[str, Any],
    ) -> None:
        self.name = self.__class__.__name__
        super(AGCRN, self).__init__()

        # Network parameters
        self.st_params = st_params
        self.dagg_params = dagg_params
        # Spatio-temporal pattern extractor
        n_layers = st_params["n_layers"]
        h_dim = st_params["h_dim"]
        cheb_k = st_params["cheb_k"]
        self.n_series = self.st_params['n_series']
        # Data Adaptive Graph Generation
        emb_dim = dagg_params["emb_dim"]
        self.out_len = out_len
        self.out_dim = out_dim

        # Model blocks
        # Node embedding matrix
        self.node_embs = nn.Parameter(torch.randn(self.n_series, emb_dim))
        # Data Adaptive Graph Generation
        self.gs_learner = AGCRNGSLearner(self.n_series, cheb_k)
        # Encoder
        self.encoder = _Encoder(
            in_dim=in_dim,
            h_dim=h_dim,
            emb_dim=emb_dim,
            n_layers=n_layers,
            cheb_k=cheb_k,
        )
        # Output layer
        self.output = nn.Conv2d(
            in_channels=1, 
            out_channels=out_len * out_dim, 
            kernel_size = (1, h_dim),
        )
        
        self._reset_parameters()
         
    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, x: Tensor, As: Optional[List[Tensor]] = None, **kwargs: Any) -> Tuple[Tuple, None, None]:
        """Forward pass.

        Args:
            x: input sequence
            As: list of adjacency matrices

        Shape:
            x: (B, P, N, C)
            output: (B, Q, N)
        """
        # Data Adaptive Graph Generation
        As = self.gs_learner(self.node_embs)
        # Encoder
        h = self.encoder(x, self.node_embs, As)   # (B, T, N, D)
        h = h[:, -1:, :, :]                       # (B, 1, N, D)

        # Output layer
        output = self.output(h).squeeze(dim=-1)           # (B, out_len * out_dim, N)
        output = output.reshape(-1, self.out_len, self.out_dim, self.n_series)
        output = output.squeeze(dim=2)   # (B, Q, N)

        return output, None, None
    
class _Encoder(nn.Module):
    """AGCRN encoder."""

    def __init__(self, in_dim: int, h_dim: int, emb_dim: int, n_layers: int, cheb_k: int) -> None:
        super(_Encoder, self).__init__()

        # Model blocks
        self.encoder = nn.ModuleList()
        for layer in range(n_layers):
            in_dim = in_dim if layer == 0 else h_dim
            self.encoder.append(
                AGCGRU(in_dim=in_dim, h_dim=h_dim, emb_dim=emb_dim, cheb_k=cheb_k)
            )

    def forward(self, x: Tensor, node_embs: Tensor, As: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: input seqeunce
            node_embs: node embeddings

        Returns:
            hs: layer-wise last hidden state

        Shape:
            x: (B, P, N, C)
            As: (cheb_k, N, N)
            hs: (B, L, N, h_dim)
        """
        hs = x
        for encoder_layer in self.encoder:
            hs, _ = encoder_layer(hs, node_embs, As, h_0=None)  # (B, L, N, h_dim)

        return hs
