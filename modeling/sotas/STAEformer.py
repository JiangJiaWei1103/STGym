"""
Baseline method, STAEformer [CIKM, 2023].
Author: ChunWei Shen

Reference:
* Paper: https://arxiv.org/abs/2308.10425
* Code: https://github.com/XDZhelheim/STAEformer
"""
from typing import List, Any, Dict, Tuple, Optional

import torch
import torch.nn as nn
from torch import Tensor

class STAEformer(nn.Module):
    def __init__(
        self,
        st_params: Dict[str, Any],
        out_len: int
    ):
        """
        STAEformer.

        Parameters:
            n_series: number of nodes
            t_window: lookback time window
            n_tids: number of times in day
            num_layers: number of self-attention layers
            input_dim: dimension of input feature
            output_dim: dimension of output
            input_embedding_dim: dimension of input embedding
            tid_embedding_dim: dimension of time in day embedding
            diw_embedding_dim: dimension of day in week embedding
            spatial_embedding_dim: dimension of spatial embedding
            adaptive_embedding_dim: dimension of adaptive embedding
            feed_forward_dim: hidden dimension of feed forward layers in self-attention
            num_heads: number of parallel attention heads
            use_mixed_proj: whether to use mixed projection
            dropout: dropout ratio
            out_len: output sequence length
        """
        super(STAEformer, self).__init__()

        # Network parameters
        self.st_params = st_params
        self.out_len = out_len
        # hyperparameters of Spatial/Temporal pattern extractor
        self.n_series = self.st_params['n_series']
        self.t_window = self.st_params['t_window']
        self.n_tids = self.st_params['n_tids']
        num_layers = self.st_params['num_layers']
        self.input_dim = self.st_params['input_dim']
        self.output_dim = self.st_params['output_dim']
        input_embedding_dim = self.st_params['input_embedding_dim']
        self.tid_embedding_dim = self.st_params['tid_embedding_dim']
        self.diw_embedding_dim = self.st_params['diw_embedding_dim']
        self.spatial_embedding_dim = self.st_params['spatial_embedding_dim']
        self.adaptive_embedding_dim = self.st_params['adaptive_embedding_dim']
        feed_forward_dim = self.st_params['feed_forward_dim']
        num_heads = self.st_params['num_heads']
        dropout = self.st_params['dropout']
        self.use_mixed_proj = self.st_params['use_mixed_proj']

        self.h_dim = (
            input_embedding_dim
            + self.tid_embedding_dim
            + self.diw_embedding_dim
            + self.spatial_embedding_dim
            + self.adaptive_embedding_dim
        )

        # input projection
        self.input_proj = nn.Linear(self.input_dim, input_embedding_dim)

        # time in day embedding
        if self.tid_embedding_dim > 0:
            self.tid_embedding = nn.Embedding(self.n_tids, self.tid_embedding_dim)

        # day in week embedding
        if self.diw_embedding_dim > 0:
            self.diw_embedding = nn.Embedding(7, self.diw_embedding_dim)

        # spatial embedding
        if self.spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(
                torch.empty(self.n_series, self.spatial_embedding_dim))
            nn.init.xavier_uniform_(self.node_emb)

        # adaptive embedding
        if self.adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(self.t_window, self.n_series, self.adaptive_embedding_dim)))

        # whether to use mixed projection or not
        if self.use_mixed_proj:
            self.output_proj = nn.Linear(
                self.t_window * self.h_dim, out_len * self.output_dim
            )
        else:
            self.temporal_proj = nn.Linear(self.t_window, out_len)
            self.output_proj = nn.Linear(self.h_dim, self.output_dim)

        # temporal self attention layers
        self.attn_layers_t = nn.ModuleList(
            [
                _SelfAttentionLayer(self.h_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        # spatial self attention layers
        self.attn_layers_s = nn.ModuleList(
            [
                _SelfAttentionLayer(self.h_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: Tensor,
        As: Optional[List[Tensor]] = None,
        **kwargs: Any,
    ) -> Tuple[Tensor, None, None]:
        """
        Forward pass.

        Parameters:
            x: input features
        
        Return:
            out: prediction

        Shape:
            x: (B, P, N, C)
            out: (B, Q, N)
        """
        batch_size = x.shape[0]

        if self.tid_embedding_dim > 0:
            tid = x[..., 1]
        if self.diw_embedding_dim > 0:
            diw = x[..., 2]

        x = x[..., :self.input_dim]

        # input projection
        x = self.input_proj(x)     # (B, T, N, input_embedding_dim)

        features = [x]
        # time in day embedding
        if self.tid_embedding_dim > 0:
            tod_emb = self.tid_embedding(
                (tid * self.n_tids).long()
            )  # (B, T, N, tid_embedding_dim)
            features.append(tod_emb)

        # day in week embedding
        if self.diw_embedding_dim > 0:
            dow_emb = self.diw_embedding(
                diw.long()
            )  # (B, T, N, diw_embedding_dim)
            features.append(dow_emb)

        # spatial embedding
        if self.spatial_embedding_dim > 0:
            spatial_emb = self.node_emb.expand(
                batch_size, self.t_window, *self.node_emb.shape
            )
            features.append(spatial_emb)

        # adaptive embedding
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(
                size=(batch_size, *self.adaptive_embedding.shape)
            )
            features.append(adp_emb)

        x = torch.cat(features, dim=-1)     # (B, T, N, h_dim)

        # temporal self attention layers
        for attn in self.attn_layers_t:
            x = attn(x, dim=1)              # (B, T, N, h_dim)
        # spatial self attention layers
        for attn in self.attn_layers_s:
            x = attn(x, dim=2)              # (B, T, N, h_dim)

        if self.use_mixed_proj:
            out = x.transpose(1, 2)         # (B, N, T, h_dim)
            out = out.reshape(batch_size, self.n_series, self.t_window * self.h_dim)
            out = self.output_proj(out).view(batch_size, self.n_series, self.out_len, self.output_dim)
            out = out.transpose(1, 2)       # (B, out_len, N, output_dim)
        else:
            out = x.transpose(1, 3)         # (B, h_dim, N, T)
            out = self.temporal_proj(out)   # (B, h_dim, N, out_len)
            out = self.output_proj(out.transpose(1, 3))  # (B, out_len, N, output_dim)

        out = out.squeeze()

        return out, None, None

class _AttentionLayer(nn.Module):
    """
    Perform attention across the -2 dim (the -1 dim is `h_dim`).

    Parameters:
        h_dim: hidden dimension of input/output
        num_heads: number of parallel attention heads
        mask: whether to mask or not
    """

    def __init__(
        self,
        h_dim: int,
        num_heads: int = 8,
        mask: bool = False
    ):
        super(_AttentionLayer, self).__init__()

        self.h_dim = h_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = h_dim // num_heads

        self.FC_Q = nn.Linear(h_dim, h_dim)
        self.FC_K = nn.Linear(h_dim, h_dim)
        self.FC_V = nn.Linear(h_dim, h_dim)

        self.out_proj = nn.Linear(h_dim, h_dim)

    def forward(
        self, 
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> Tensor:
        """
        Forward pass.

        Parameters:
            query: query embeddings for attention
            key: key embeddings for attention
            value: value embeddings for attention
        
        Shape:
            query: (B, ..., tgt_length, D)
            key: (B, ..., src_length, D)
            value: (B, ..., src_length, D)
            out: (B, ..., tgt_length, D)
        """
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # (num_heads * B, ..., tgt_length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        # (num_heads * B, ..., src_length, head_dim)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        # (num_heads * B, ..., src_length, head_dim)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(-1, -2)  # (num_heads * B, ..., head_dim, src_length)

        attn_score = (query @ key) / self.head_dim**0.5  # (num_heads * B, ..., tgt_length, src_length)

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * B, ..., tgt_length, head_dim)
        out = torch.cat(torch.split(out, batch_size, dim=0), dim=-1)  # (B, ..., tgt_length, h_dim)

        out = self.out_proj(out)

        return out

class _SelfAttentionLayer(nn.Module):
    def __init__(
        self,
        h_dim: int,
        feed_forward_dim: int = 2048,
        num_heads: int = 8,
        dropout: float = 0,
        mask: bool = False
    ):
        """
        Self-Attention Layer.

        Parameters:
            h_dim: hidden dimension of attention input/output
            feed_forward_dim: hidden dimension of feed forward layers
            num_heads: number of parallel attention heads
            dropout: dropout ratio
            mask: whether to mask or not
        """
        super(_SelfAttentionLayer, self).__init__()

        self.attn = _AttentionLayer(h_dim, num_heads, mask)

        self.feed_forward = nn.Sequential(
            nn.Linear(h_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, h_dim),
        )

        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        dim: int = -2
    ) -> Tensor:
        """
        Forward pass.

        Parameters:
            x: input feature
            dim: which dimension to perform attention

        Shape:
            x: (B, T, N, D)
            out: (B, T, N, D)
        """
        x = x.transpose(dim, -2)        # (B, ..., length, h_dim)

        residual = x
        out = self.attn(x, x, x)        # (B, ..., length, h_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)    # (B, ..., length, h_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)

        return out