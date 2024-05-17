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
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        bias: bool = True,
        bn: Optional[bool] = False,
        act: Optional[str] = None, 
        dropout: Optional[float] = None
    ) -> None:
        super(Linear2d, self).__init__()

        # Model blocks
        self.lin = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=(1, 1), bias=bias)

        if bn:
            self.bn = nn.BatchNorm2d(out_features)
        else:
            self.bn = None
        
        if act is not None:
            if act == "relu":
                self.act = nn.ReLU()
            elif act == "tanh":
                self.act = nn.Tanh()
            elif act == "sigmoid":
                self.act = nn.Sigmoid()
        else:
            self.act = None
        
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters:
            x: input

        Return:
            output: output

        Shape:
            x: (B, in_features, H, W)
            output: (B, out_features, H, W)
        """
        output = self.lin(x)

        if self.bn is not None:
            output = self.bn(output)

        if self.act is not None:
            output = self.act(output)
        
        if self.dropout is not None:
            output = self.dropout(output)

        return output


class MultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron."""
    def __init__(
        self, 
        in_dims: List[int], 
        h_dims: List[int],
        acts: List[str],
        dropouts: List[float],
        bns: List[bool],
        residual: bool = False
    ):
        super(MultiLayerPerceptron, self).__init__()

        # Network parameters
        self.residual = residual

        # Model blocks
        self.mlp_layers = nn.ModuleList()
        for in_dim, h_dim, bn, act, dropout in zip(in_dims, h_dims, bns, acts, dropouts):
            self.mlp_layers.append(
                Linear2d(
                    in_features=in_dim,
                    out_features=h_dim,
                    bn=bn,
                    act=act,
                    dropout=dropout
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters:
            x: input

        Return:
            h: hidden output

        Shape:
            x: (B, in_features, N, L)
            h: (B, out_features, N, L)
        """
        h = x
        for layer in range(len(self.mlp_layers)):
            h = self.mlp_layers[layer](h)

        if self.residual:
            h = h + x

        return h


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
            tid_emb_init = nn.Parameter(torch.empty(n_tids, tid_emb_dim))
            nn.init.xavier_uniform_(tid_emb_init)
            self.tid_emb = nn.Embedding.from_pretrained(tid_emb_init, freeze=False)
        if diw_emb_dim > 0:
            diw_emb_init = nn.Parameter(torch.empty(N_DAYS_IN_WEEK, diw_emb_dim))
            nn.init.xavier_uniform_(diw_emb_init)
            self.diw_emb = nn.Embedding.from_pretrained(diw_emb_init, freeze=False)
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
            x_tid = self.tid_emb(tid)    # (B, N, tid_emb_dim) or (B, T, N, tid_emb_dim)
        if self.diw_emb is not None:
            assert diw is not None, "Day in week isn't fed into the model."
            x_diw = self.diw_emb(diw)    # (B, N, diw_emb_dim) or (B, T, N, diw_emb_dim)
        if self.adp_emb is not None:
            x_adp = self.adp_emb         # (N, T, adp_emb_dim)

        return x_node_in, x_node_out, x_tid, x_diw, x_adp


class AttentionLayer(nn.Module):
    """Attention Layer.

    Perform attention across the -2 dim (the -1 dim is `h_dim`).
    """
    def __init__(
        self, in_dim: int, h_dim: int, n_heads: int, mask: bool = False, act: str = None, bn: bool = False
    ) -> None:
        super(AttentionLayer, self).__init__()

        # Network parameters
        self.mask = mask
        self.head_dim = h_dim // n_heads

        # Model blocks
        self.FC_Q = Linear2d(in_features=in_dim, out_features=h_dim, bn=bn, act=act)
        self.FC_K = Linear2d(in_features=in_dim, out_features=h_dim, bn=bn, act=act)
        self.FC_V = Linear2d(in_features=in_dim, out_features=h_dim, bn=bn, act=act)
        self.output = nn.Linear(h_dim, h_dim)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """Forward pass.

        Parameters:
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

        output = self.output(output)

        return output
    

class Memory(nn.Module):
    """Construct memory."""

    def __init__(
        self,
        mem_num: int,
        mem_dim: int,
        h_dim: int,
        n_series: int,
    ):
        super(Memory, self).__init__()

        # Model blocks
        self.Memory = nn.Parameter(torch.randn(mem_num, mem_dim))   # (M, d)
        self.Wq = nn.Parameter(torch.randn(h_dim, mem_dim))         # project to query
        self.We1 = nn.Parameter(torch.randn(n_series, mem_num))     # project memory to embedding
        self.We2 = nn.Parameter(torch.randn(n_series, mem_num))     # project memory to embedding

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_normal_(self.Memory)
        nn.init.xavier_normal_(self.Wq)
        nn.init.xavier_normal_(self.We1)
        nn.init.xavier_normal_(self.We2)

    def forward(self, x: Optional[Tensor] = None) -> Tensor:
        """Forward pass.

        Parameters:
            x: input

        Shape:
            x: (B, N, C)
        """

        if x is None:
            return self.Memory, self.We1, self.We2
        else:
            # Query memory
            query = torch.matmul(x, self.Wq)     # (B, N, d)
            att_score = torch.softmax(torch.matmul(query, self.Memory.t()), dim=-1)   # (B, N, M)
            value = torch.matmul(att_score, self.Memory)     # (B, N, d)
            _, ind = torch.topk(att_score, k=2, dim=-1)
            pos = self.Memory[ind[:, :, 0]]                  # (B, N, d)
            neg = self.Memory[ind[:, :, 1]]                  # (B, N, d)

            return value, query, pos, neg


class SNorm(nn.Module):
    """Spatial Normalization."""
    def __init__(self, in_dim: int) -> None:
        super(SNorm, self).__init__()

        # Model blocks
        self.beta = nn.Parameter(torch.zeros(in_dim))
        self.gamma = nn.Parameter(torch.ones(in_dim))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters:
            x: input

        Return:
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

        Parameters:
            x: input

        Return:
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


class Split(nn.Module):
    """
    Downsamples the original sequence into two sub-sequences
    by separating the even and the odd elements.
    """
    def __init__(self) -> None:
        super(Split, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters:
            x: input

        Shape:
            x: (B, L, N)
        """
        x_even = x[:, ::2, :]
        x_odd = x[:, 1::2, :]

        return x_even, x_odd


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

        Parameters:
            x: input

        Return:
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