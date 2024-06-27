"""
Baseline method, STAEformer [CIKM, 2023].
Author: ChunWei Shen

Reference:
* Paper: https://arxiv.org/abs/2308.10425
* Code: https://github.com/XDZhelheim/STAEformer
"""
from typing import List, Any, Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from metadata import N_DAYS_IN_WEEK

from modeling.module.layers import STAEAttentionLayer
from modeling.module.common_layers import AuxInfoEmbeddings

class STAEformer(nn.Module):
    """STAEformer framework.

        Args:
            in_dim: input feature dimension
            in_len: input sequence length
            out_dim: output dimension
            out_len: output sequence length
            n_series: number of series
            n_layers: number of Attention layers
            lin_h_dim: hidden dimension of input linear layer
            n_heads: number of parallel attention heads
            tid_emb_dim: dimension of time in day embedding
            diw_emb_dim: dimension of day in week embedding
            adp_emb_dim: dimension of adaptive embedding
            node_emb_dim: dimension of static node embedding
            n_tids: number of times in day
            use_mixed_proj: if Ture, output layer use mixed projection
            dropout: dropout ratio
    """
    
    def __init__(
        self, in_dim: int, in_len: int, out_dim: int, out_len: int, n_series: int, st_params: Dict[str, Any]
    ) -> None:
        self.name = self.__class__.__name__
        super(STAEformer, self).__init__()

        # Network parameters
        self.st_params = st_params
        # Spatio-temporal pattern extractor
        n_layers = st_params["n_layers"]
        lin_h_dim = st_params["lin_h_dim"]
        ffl_h_dim = st_params["ffl_h_dim"]
        n_heads = st_params["n_heads"]
        self.node_emb_dim = st_params["node_emb_dim"]
        self.tid_emb_dim = st_params["tid_emb_dim"]
        self.diw_emb_dim = st_params["diw_emb_dim"]
        self.adp_emb_dim = st_params["adp_emb_dim"]
        self.n_tids = st_params["n_tids"]
        self.use_mixed_proj = st_params["use_mixed_proj"]
        dropout = st_params["dropout"]
        self.in_dim = in_dim
        self.in_len = in_len
        self.out_dim = out_dim
        self.out_len = out_len
        self.n_series = n_series
        self.h_dim = (
            lin_h_dim
            + self.node_emb_dim
            + self.tid_emb_dim
            + self.diw_emb_dim
            + self.adp_emb_dim
        )

        # Model blocks
        # Input linear layer
        self.in_lin = nn.Linear(in_dim, lin_h_dim)
        # Auxiliary information embeddings
        self.aux_info_emb = AuxInfoEmbeddings(
            n_series=n_series,
            t_window=in_len,
            node_emb_dim=self.node_emb_dim,
            adp_emb_dim=self.adp_emb_dim,
        )
        if self.tid_emb_dim > 0:
            self.tid_emb = nn.Embedding(self.n_tids, self.tid_emb_dim)
        if self.diw_emb_dim > 0:
            self.diw_emb = nn.Embedding(N_DAYS_IN_WEEK, self.diw_emb_dim)
        # Temporal attention layers
        self.temporal_attn_layers = nn.ModuleList(
            [
                STAEAttentionLayer(
                    in_dim=self.h_dim, h_dim=self.h_dim, ffl_h_dim=ffl_h_dim, n_heads=n_heads, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )
        # Spatial attention layers
        self.spatial_attn_layers = nn.ModuleList(
            [
                STAEAttentionLayer(
                    in_dim=self.h_dim, h_dim=self.h_dim, ffl_h_dim=ffl_h_dim, n_heads=n_heads, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )
        # Output layer
        if self.use_mixed_proj:
            self.output = nn.Linear(in_len * self.h_dim, out_len * out_dim)
        else:
            self.output_temporal = nn.Linear(in_len, out_len)
            self.output = nn.Linear(self.h_dim, out_dim)

    def forward(self, x: Tensor, As: List[Tensor] = None, **kwargs: Any) -> Tuple[Tensor, None, None]:
        """Forward pass.

        Args:
            x: input sequence

        Returns:
            output: prediction

        Shape:
            x: (B, P, N, C)
            output: (B, Q, N)
        """
        batch_size = x.shape[0]
        if self.tid_emb_dim > 0:
            tid = (x[..., 1] * self.n_tids).long()
        if self.diw_emb_dim > 0:
            diw = (x[..., 2] * N_DAYS_IN_WEEK).long()
        x = x[..., :self.in_dim]

        # Input linear layer
        x = self.in_lin(x)     # (B, T, N, lin_h_dim)
        # Auxiliary information embeddings
        features = [x]
        x_node, _, _, _, x_adp = self.aux_info_emb()
        # Time in day embedding
        if self.tid_emb_dim > 0:
            x_tid = self.tid_emb(tid)
            features.append(x_tid)
        # Day in week embedding
        if self.diw_emb_dim > 0:
            x_diw = self.diw_emb(diw)
            features.append(x_diw)
        # Node embedding
        if self.node_emb_dim > 0:
            features.append(x_node.expand(batch_size, self.in_len, *x_node.shape))
        # Adaptive embedding
        if self.adp_emb_dim > 0:
            features.append(x_adp.expand(batch_size, *x_adp.shape))
    
        h = torch.cat(features, dim=-1)     # (B, T, N, h_dim)

        # Temporal self attention layers
        for attn_layer in self.temporal_attn_layers:
            h = attn_layer(h, dim=1)        # (B, T, N, h_dim)
        # Spatial self attention layers
        for attn_layer in self.spatial_attn_layers:
            h = attn_layer(h, dim=2)        # (B, T, N, h_dim)

        # Output layer
        if self.use_mixed_proj:
            h = h.transpose(1, 2)         # (B, N, T, h_dim)
            h = h.reshape(batch_size, self.n_series, self.in_len * self.h_dim)
            h = self.output(h).view(batch_size, self.n_series, self.out_len, self.out_dim)
            output = h.transpose(1, 2).squeeze(dim=-1)   # (B, Q, N)
        else:
            h = x.transpose(1, 3)                        # (B, h_dim, N, T)
            h = self.output_temporal(h)                  # (B, h_dim, N, out_len)
            output = self.output(h.transpose(1, 3)).squeeze(dim=-1)      # (B, Q, N)

        return output, None, None