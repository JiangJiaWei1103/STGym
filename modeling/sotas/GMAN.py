"""
Baseline method, GMAN [AAAI, 2020].
Author: ChunWei Shen

Reference:
* Paper: https://arxiv.org/abs/1911.08415
* Code: 
    * https://github.com/zhengchuanpan/GMAN
    * https://github.com/benedekrozemberczki/pytorch_geometric_temporal
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from metadata import N_DAYS_IN_WEEK

from modeling.module.common_layers import Linear2d, SpatioTemporalEmbedding, AttentionLayer
from modeling.module.layers import GMANAttentionBlock

class GMAN(nn.Module):
    """GMAN framework.

    Args:
        in_len: input sequence length
        in_dim: input feature dimension
        out_dim: output dimension
        n_layers: number of attention layers
        n_heads: number of attention heads
        h_dim: hidden dimension
        n_tids: number of time slots in one day
    """

    def __init__(
        self, 
        in_len: int, 
        in_dim: int, 
        out_dim: int,
        st_params: Dict[str, Any],
    ) -> None:
        super(GMAN, self).__init__()

        # Network parameters
        self.st_params = st_params
        # Spatio-temporal pattern extractor
        n_layers = st_params["n_layers"]
        n_heads = st_params["n_heads"]
        h_dim = st_params["h_dim"]
        self.n_tids = st_params["n_tids"]
        self.in_len = in_len

        # Model blocks
        # Input linear layer
        self.in_lin = nn.Sequential(
            Linear2d(in_dim, h_dim, reset_params=True),
            nn.BatchNorm2d(h_dim),
            nn.ReLU(),
            Linear2d(h_dim, h_dim, reset_params=True),
            nn.BatchNorm2d(h_dim)
        )
        # Spatio Temporal Embedding
        self.st_embedding = SpatioTemporalEmbedding(h_dim=h_dim, n_tids=self.n_tids)
        # Encoder
        self.encoder = nn.ModuleList(
            [GMANAttentionBlock(in_dim=h_dim, h_dim=h_dim, n_heads=n_heads) for _ in range(n_layers)]
        )
        # Decoder
        self.decoder = nn.ModuleList(
            [GMANAttentionBlock(in_dim=h_dim, h_dim=h_dim, n_heads=n_heads) for _ in range(n_layers)]
        )
        # Transform Attention
        self.transform_attention = AttentionLayer(in_dim=h_dim, h_dim=h_dim, n_heads=n_heads, act="relu", bn=True)
        # Output layer
        self.output = nn.Sequential(
            Linear2d(h_dim, h_dim, reset_params=True),
            nn.BatchNorm2d(h_dim),
            nn.ReLU(),
            Linear2d(h_dim, out_dim, reset_params=True),
            nn.BatchNorm2d(out_dim)
        )

    def forward(
        self, x: Tensor, As: Optional[List[Tensor]] = None, ycl: Tensor = None, aux_data: Tensor = None, **kwargs: Any
    ) -> Tuple[Tensor, None, None]:
        """Forward pass.

        Args:
            x: input sequence

        Returns:
            output: prediction

        Shape:
            x: (B, P, N, C)
            output: (B, Q, N)
        """
        x_tid = x[:, :, 0, 1] * self.n_tids
        x_diw = x[:, :, 0, 2] * N_DAYS_IN_WEEK
        y_tid = ycl[:, :, 0, 1] * self.n_tids
        y_diw = ycl[:, :, 0, 2] * N_DAYS_IN_WEEK

        x_temporal_emb = torch.cat((x_diw.unsqueeze(-1), x_tid.unsqueeze(-1)), dim=-1)
        y_temporal_emb = torch.cat((y_diw.unsqueeze(-1), y_tid.unsqueeze(-1)), dim=-1)
        temporal_emb = torch.cat((x_temporal_emb, y_temporal_emb), dim=1).type(torch.int32)
 
        x = x[..., 0].unsqueeze(-1).transpose(1, 3)     # (B, C, N, P)
        # Input linear layer
        x = self.in_lin(x).transpose(1, 3)              # (B, P, N, C')
        # Spatio Temporal Embedding
        spatial_emb = torch.Tensor(aux_data[0]).to(x.device)
        st_emb = self.st_embedding(spatial_emb, temporal_emb, self.n_tids)
        st_emb_his = st_emb[:, :self.in_len]
        st_emb_pred = st_emb[:, self.in_len:]
        # Encoder
        for attn_layer in self.encoder:
            x = attn_layer(x, st_emb_his)
        # Transform Attention
        x = self.transform_attention(st_emb_pred.transpose(1, 2), st_emb_his.transpose(1, 2), x.transpose(1, 2))
        x = x.transpose(1, 2)
        # Decoder
        for attn_layer in self.decoder:
            x = attn_layer(x, st_emb_pred)

        output = self.output(x.transpose(1, 3))
        output = output.squeeze(dim=1).transpose(1, 2)

        return output, None, None