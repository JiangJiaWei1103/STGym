"""
Baseline method, STID [CIKM, 2022].
Author: ChunWei Shen

Reference:
* Paper: https://arxiv.org/abs/2208.05233
* Code: https://github.com/zezhishao/STID
"""
from typing import List, Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from metadata import N_DAYS_IN_WEEK

from modeling.module.common_layers import Linear2d, AuxInfoEmbeddings

class STID(nn.Module):
    """STID framework.

    Args:
        in_dim: input feature dimension
        in_len: input sequence length
        out_len: output sequence length
        n_series: number of series
        n_layers: number of MLP layers
        lin_h_dim: hidden dimension of input linear layer
        node_emb_dim: dimension of static node embedding
        tid_emb_dim: dimension of time in day embedding
        diw_emb_dim: dimension of day in week embedding
        n_tids: number of times in day
    """

    def __init__(self, in_dim: int, in_len: int, out_len: int, n_series: int, st_params: Dict[str, Any]) -> None:
        self.name = self.__class__.__name__
        super(STID, self).__init__()

        # Network parameters
        self.st_params = st_params
        # Spatio-temporal pattern extractor
        self.n_layers = st_params["n_layers"]
        lin_h_dim = st_params["lin_h_dim"]
        self.node_emb_dim = st_params["node_emb_dim"]
        self.tid_emb_dim = st_params["tid_emb_dim"]
        self.diw_emb_dim = st_params["diw_emb_dim"]
        self.n_tids = st_params["n_tids"]
        self.in_dim = in_dim

        # Model blocks
        # Input linear layer
        self.in_lin = Linear2d(in_dim * in_len, lin_h_dim)
        # Auxiliary information embeddings
        self.aux_info_emb = AuxInfoEmbeddings(
            n_tids=self.n_tids,
            n_series=n_series,
            node_emb_dim=self.node_emb_dim,
            tid_emb_dim=self.tid_emb_dim,
            diw_emb_dim=self.diw_emb_dim
        )
        # Encoder
        h_dim = lin_h_dim + self.node_emb_dim + self.tid_emb_dim + self.diw_emb_dim
        self.encoder = nn.Sequential(
            *[
                MultiLayerPerceptron(in_dim=h_dim, h_dim=h_dim) for _ in range(self.n_layers)
            ]
        )
        # Output layer
        self.output = Linear2d(in_features=h_dim, out_features=out_len)

    def forward(self, x: Tensor, As: Optional[List[Tensor]] = None, **kwargs: Any) -> Tuple[Tensor, None, None]:
        """Forward pass.

        Args:
            x: input sequence

        Returns:
            output: prediction

        Shape:
            x: (B, P, N, C)
            output: (B, Q, N)
        """
        if self.tid_emb_dim > 0:
            tid = (x[:, -1, :, 1] * self.n_tids).long()
        if self.diw_emb_dim > 0:
            diw = (x[:, -1, :, 2] * N_DAYS_IN_WEEK).long()
        x = x[..., range(self.in_dim)]
        
        # Input linear layer
        batch_size, _, n_series, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, n_series, -1).transpose(1, 2).unsqueeze(-1)
        h = self.in_lin(x)

        # Auxiliary information embeddings
        embs = []
        x_node, _, x_tid, x_diw, _ = self.aux_info_emb(tid=tid, diw=diw)
        if self.node_emb_dim > 0:
            embs.append(x_node.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2).unsqueeze(-1))
        if self.tid_emb_dim > 0:
            embs.append(x_tid.transpose(1, 2).unsqueeze(-1))
        if self.diw_emb_dim > 0:
            embs.append(x_diw.transpose(1, 2).unsqueeze(-1))
        h = torch.cat([h] + embs, dim=1)    # concate all embeddings along channel

        # Encoder
        h = self.encoder(h)   # (B, D, N, 1)

        # Output layer
        output = self.output(h).squeeze(-1)  # (B, Q, N)

        return output, None, None

class MultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, in_dim: int, h_dim: int) -> None:
        super(MultiLayerPerceptron, self).__init__()

        self.mlp = nn.Sequential(
            Linear2d(in_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            Linear2d(h_dim, h_dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: input 
            
        Shape:
            x: (B, in_dim, N, 1)
            output: (B, h_dim, N, 1)
        """

        hidden = self.mlp(x)
        output = hidden + x

        return output