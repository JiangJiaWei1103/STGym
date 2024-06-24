"""
Common-layers.
Author: JiaWei Jiang, ChunWei Shen
"""
from typing import List, Optional, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from metadata import N_DAYS_IN_WEEK


# Common
class Linear2d(nn.Module):
    """Linear layer over 2D plane.

    Linear2d applies linear transformation along channel dimension of
    2D planes.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, reset_params: bool = False) -> None:
        super(Linear2d, self).__init__()

        # Network parameters
        self.bias = bias
        # Model blocks
        self.lin = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=(1, 1), bias=bias)

        if reset_params:
            self._reset_parameters()

    def _reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.lin.weight)
        if self.bias:
            torch.nn.init.zeros_(self.lin.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: input

        Returns:
            output: output

        Shape:
            x: (B, in_features, H, W)
            output: (B, out_features, H, W)
        """
        output = self.lin(x)

        return output
    

class AttentionLayer(nn.Module):
    """Attention Layer.

    Perform attention across the -2 dim (the -1 dim is `h_dim`).
    """

    def __init__(
        self, 
        in_dim: int, 
        h_dim: int, 
        n_heads: int, 
        mask: bool = False, 
        act: str = None, 
        bn: bool = False, 
        reset_params: bool=False
    ) -> None:
        super(AttentionLayer, self).__init__()

        # Network parameters
        self.bn = bn
        self.mask = mask
        self.head_dim = h_dim // n_heads

        # Model blocks
        FC_Q = [Linear2d(in_features=in_dim, out_features=h_dim, reset_params=reset_params)]
        FC_K = [Linear2d(in_features=in_dim, out_features=h_dim, reset_params=reset_params)]
        FC_V = [Linear2d(in_features=in_dim, out_features=h_dim, reset_params=reset_params)]
        output = [Linear2d(in_features=h_dim, out_features=h_dim, reset_params=reset_params)]

        if bn:
            FC_Q.append(nn.BatchNorm2d(h_dim))
            FC_K.append(nn.BatchNorm2d(h_dim))
            FC_V.append(nn.BatchNorm2d(h_dim))
            output.append(nn.BatchNorm2d(h_dim))

        self.act = None
        if act is not None:
            if act == "relu":
                FC_Q.append(nn.ReLU())
                FC_K.append(nn.ReLU())
                FC_V.append(nn.ReLU())
                output.append(nn.ReLU())

        self.FC_Q = nn.Sequential(*FC_Q)
        self.FC_K = nn.Sequential(*FC_K)
        self.FC_V = nn.Sequential(*FC_V)
        self.output = nn.Sequential(*output)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """Forward pass.

        Args:
            query: query embeddings for attention
            key: key embeddings for attention
            value: value embeddings for attention
        
        Shape:
            query: (B, ..., tgt_length, in_dim)
            key: (B, ..., src_length, in_dim)
            value: (B, ..., src_length, in_dim)
            output: (B, ..., tgt_length, h_dim)
        """
        batch_size, _, tgt_length, _ = query.shape
        src_length = key.shape[-2]

        query = self.FC_Q(query.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        key = self.FC_K(key.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        value = self.FC_V(value.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)

        # (num_heads * B, ..., tgt_length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        # (num_heads * B, ..., src_length, head_dim)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        # (num_heads * B, ..., src_length, head_dim)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)
        # (num_heads * B, ..., head_dim, src_length)
        key = key.transpose(-1, -2)  
        # (num_heads * B, ..., tgt_length, src_length)
        attn_score = (query @ key) / self.head_dim**0.5  

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        output = attn_score @ value  # (num_heads * B, ..., tgt_length, head_dim)
        output = torch.cat(torch.split(output, batch_size, dim=0), dim=-1)  # (B, ..., tgt_length, h_dim)

        output = self.output(output.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)

        return output


class SpatioTemporalEmbedding(nn.Module):
    """Spatial-Temporal Embedding block."""

    def __init__(self, h_dim: int, n_tids: int) -> None:
        super(SpatioTemporalEmbedding, self).__init__()

        # Model blocks
        self.linear_se = nn.Sequential(
            Linear2d(h_dim, h_dim, reset_params=True),
            nn.BatchNorm2d(h_dim),
            nn.ReLU(),
            Linear2d(h_dim, h_dim, reset_params=True),
            nn.BatchNorm2d(h_dim)
        )
        self.linear_te = nn.Sequential(
            Linear2d(n_tids + 7, h_dim, reset_params=True),
            nn.BatchNorm2d(h_dim),
            nn.ReLU(),
            Linear2d(h_dim, h_dim, reset_params=True),
            nn.BatchNorm2d(h_dim)
        )

    def forward(self, spatial_emb: Tensor, temporal_emb: Tensor, n_tids: int) -> Tensor:
        """Forward pass.

        Args:
            spatial_emb: Spatial embedding
            temporal_emb: Temporal embedding
            n_tids: number of time slots in one day

        Returns:
            output: Spatial-Temporal embedding

        Shape:
            spatial_emb: (N, D)
            temporal_emb: (B, in_len + out_len, 2)
            output: (B, in_len + out_len, N ,D)
        """
        batch_size, seq_len, _ = temporal_emb.shape
        # Spatial embedding
        spatial_emb = spatial_emb.expand(1, 1, -1, -1).transpose(1, 3)   # (1, D, N, 1)
        spatial_emb = self.linear_se(spatial_emb).transpose(1, 3)        # (1, 1, N, D)
        # Temporal embedding
        diw = torch.empty(batch_size, seq_len, 7).to(temporal_emb.device)
        tid = torch.empty(batch_size, seq_len, n_tids).to(temporal_emb.device)
        for i in range(batch_size):
            diw[i] = F.one_hot(temporal_emb[..., 0][i].to(torch.int64) % 7, 7)
            tid[i] = F.one_hot(temporal_emb[..., 1][i].to(torch.int64) % n_tids, n_tids)
        temporal_emb = torch.cat((diw, tid), dim=-1)
        temporal_emb = temporal_emb.unsqueeze(dim=2).transpose(1, 3)  # (B, n_tids + 7, 1, in_len + out_len)
        temporal_emb = self.linear_te(temporal_emb).transpose(1, 3)   # (B, in_len + out_len, 1, D)

        output = spatial_emb + temporal_emb                           # (B, in_len + out_len, N, D)

        return output
    

class GatedFusion(nn.Module):
    """Gated fusion mechanism. """

    def __init__(self, h_dim: int) -> None:
        super(GatedFusion, self).__init__()

        # Model blocks
        self.linear_s = nn.Sequential(Linear2d(h_dim, h_dim, bias=False, reset_params=True), nn.BatchNorm2d(h_dim))
        self.linear_t = nn.Sequential(Linear2d(h_dim, h_dim, reset_params=True), nn.BatchNorm2d(h_dim))
        self.output = nn.Sequential(
            Linear2d(h_dim, h_dim, reset_params=True),
            nn.BatchNorm2d(h_dim),
            nn.ReLU(),
            Linear2d(h_dim, h_dim, reset_params=True),
            nn.BatchNorm2d(h_dim),
        )

    def forward(self, hs: Tensor, ht: Tensor) -> Tensor:
        """Forward pass.

        Args:
            hs: hidden state of spatial attention
            ht: hidden state of temporal attention
        
        Shape:
            hs: (B, L, N, h_dim)
            ht: (B, L, N, h_dim)
            output: (B, L, N, h_dim)
        """
        hs_l = self.linear_s(hs.transpose(1, 3)).transpose(1, 3)     # (B, L, N, h_dim)
        ht_l = self.linear_t(ht.transpose(1, 3)).transpose(1, 3)     # (B, L, N, h_dim)
        z = torch.sigmoid(hs_l + ht_l)                               # (B, L, N, h_dim)
        h = torch.mul(z, hs) + torch.mul(1 - z, ht)                  # (B, L, N, h_dim)
        output = self.output(h.transpose(1, 3)).transpose(1, 3)      # (B, L, N, h_dim)

        return output


class AuxInfoEmbeddings(nn.Module):
    """Auxiliary information embeddings."""
    
    def __init__(
        self,
        n_tids: Optional[int] = None,
        n_series: Optional[int] = None,
        t_window: Optional[int] = None,
        node_emb_dim: Optional[int] = 0,
        tid_emb_dim: Optional[int] = 0,
        diw_emb_dim: Optional[int] = 0,
        adp_emb_dim: Optional[int] = 0,
        single_node_emb: Optional[bool] = True,
    ) -> None:
        super(AuxInfoEmbeddings, self).__init__()

        # Network parameters
        self.n_series = n_series
        self.node_emb_dim = node_emb_dim
        self.tid_emb_dim = tid_emb_dim
        self.diw_emb_dim = diw_emb_dim
        self.adp_emb_dim = adp_emb_dim
        self.single_node_emb = single_node_emb

        # Model blocks
        self.node_emb_in, self.node_emb_out, self.tid_emb, self.diw_emb, self.adp_emb = None, None, None, None, None
        if node_emb_dim > 0:
            self.node_emb_in = nn.Parameter(torch.empty(n_series, node_emb_dim))
            nn.init.xavier_uniform_(self.node_emb_in)
            if not single_node_emb:
                self.node_emb_out = nn.Parameter(torch.empty(n_series, node_emb_dim))
                nn.init.xavier_uniform_(self.node_emb_out)
        if tid_emb_dim > 0:
            self.tid_emb = nn.Parameter(torch.empty(n_tids, tid_emb_dim))
            nn.init.xavier_uniform_(self.tid_emb)
        if diw_emb_dim > 0:
            self.diw_emb = nn.Parameter(torch.empty(N_DAYS_IN_WEEK, diw_emb_dim))
            nn.init.xavier_uniform_(self.diw_emb)
        if adp_emb_dim > 0:
            self.adp_emb = nn.Parameter(torch.empty(t_window, n_series, adp_emb_dim))
            nn.init.xavier_uniform_(self.adp_emb)

    def forward(self, tid: Optional[Tensor] = None, diw: Optional[Tensor] = None) -> Tensor:
        """Forward pass.

        Shape:
            tid: (B, N) or (B, T, N)
            diw: (B, N) or (B, T, N)
        """
        x_node_in, x_node_out, x_tid, x_diw, x_adp = None, None, None, None, None

        if self.node_emb_in is not None:
            x_node_in = self.node_emb_in       # (N, node_emb_dim)
            if not self.single_node_emb:
                x_node_out = self.node_emb_out
        if self.tid_emb is not None:
            assert tid is not None, "Time in day isn't fed into the model."
            x_tid = self.tid_emb[tid]    # (B, N, tid_emb_dim) or (B, T, N, tid_emb_dim)
        if self.diw_emb is not None:
            assert diw is not None, "Day in week isn't fed into the model."
            x_diw = self.diw_emb[diw]    # (B, N, diw_emb_dim) or (B, T, N, diw_emb_dim)
        if self.adp_emb is not None:
            x_adp = self.adp_emb         # (N, T, adp_emb_dim)

        return x_node_in, x_node_out, x_tid, x_diw, x_adp


class SNorm(nn.Module):
    """Spatial Normalization."""

    def __init__(self, in_dim: int) -> None:
        super(SNorm, self).__init__()

        # Model blocks
        self.beta = nn.Parameter(torch.zeros(in_dim))
        self.gamma = nn.Parameter(torch.ones(in_dim))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: input

        Returns:
            output: output

        Shape:
            x: (B, in_dim, N, L)
            output: (B, in_dim, N, L)
        """
        x_norm = (x - x.mean(2, keepdims=True)) / (x.var(2, keepdims=True, unbiased=True) + 0.00001) ** 0.5
        output = x_norm * self.gamma.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)

        return output


class TNorm(nn.Module):
    """Temporal Normalization.""" 

    def __init__(
        self, in_dim: int, n_series: int, track_running_stats: bool = True, momentum: float = 0.1
    ) -> None:
        super(TNorm, self).__init__()

        # Network parameters
        self.track_running_stats = track_running_stats
        self.momentum = momentum

        # Model blocks
        self.beta = nn.Parameter(torch.zeros(1, in_dim, n_series, 1))
        self.gamma = nn.Parameter(torch.ones(1, in_dim, n_series, 1))
        self.register_buffer('running_mean', torch.zeros(1, in_dim, n_series, 1))
        self.register_buffer('running_var', torch.ones(1, in_dim, n_series, 1))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: input

        Returns:
            output: output

        Shape:
            x: (B, in_dim, N, L)
            output: (B, in_dim, N, L)
        """
        if self.track_running_stats:
            mean = x.mean((0, 3), keepdims=True)
            var = x.var((0, 3), keepdims=True, unbiased=False)
            if self.training:
                n = x.shape[3] * x.shape[0]
                with torch.no_grad():
                    self.running_mean = self.momentum * mean + (1 - self.momentum) * self.running_mean
                    self.running_var = self.momentum * var * n / (n - 1) + (1 - self.momentum) * self.running_var
            else:
                mean = self.running_mean
                var = self.running_var
        else:
            mean = x.mean((3), keepdims=True)
            var = x.var((3), keepdims=True, unbiased=True)

        x_norm = (x - mean) / (var + 0.00001) ** 0.5
        output = x_norm * self.gamma + self.beta

        return output


class Align(nn.Module):
    """
    Ensure alignment of input feature dimensions for 
    the residual connection.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super(Align, self).__init__()

        # Network parameters
        self.in_features = in_features
        self.out_features = out_features

        # Model blocks
        self.align_conv = Linear2d(in_features=in_features, out_features=out_features)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: input

        Returns:
            output: output

        Shape:
            x: (B, in_features,  N, L)
            output: (B, out_features,  N, L)
        """
        batch_size, _, n_series, t_window = x.shape

        if self.in_features > self.out_features:
            output = self.align_conv(x)
        elif self.in_features < self.out_features:
            zeros = torch.zeros([batch_size, self.out_features - self.in_features, n_series, t_window]).to(x.device)
            output = torch.cat([x, zeros], dim = 1)
        else:
            output = x
        
        return output