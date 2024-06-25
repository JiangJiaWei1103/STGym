"""
Common sptio-temporal layers.
Author: JiaWei Jiang, ChunWei Shen
"""
from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .tconv import TConvBaseModule

from .common_layers import Linear2d, AttentionLayer, GatedFusion
from .temporal_layers import GatedTCN, DilatedInception, TemporalConvLayer, SCIBlock
from .spatial_layers import DiffusionConvLayer, ChebGraphConv, InfoPropLayer, STSGCM, NAPLConvLayer

class DCGRU(nn.Module):
    """Diffusion convolutional gated recurrent unit.

    Args:
        in_dim: input feature dimension
        h_dim: hidden state dimension
        n_adjs: number of adjacency matrices
            *Note: Bidirectional transition matrices are used in the
                original proposal.
        max_diffusion_step: maximum diffusion step
    """

    def __init__(
        self, 
        in_dim: int, 
        h_dim: int, 
        n_adjs: int = 2, 
        max_diffusion_step: int = 2, 
    ) -> None:
        super(DCGRU, self).__init__()

        # Network parameters
        self.h_dim = h_dim

        # Model blocks
        cat_dim = in_dim + h_dim
        self.gate = DiffusionConvLayer(
            in_dim=cat_dim, h_dim=h_dim * 2, n_adjs=n_adjs, max_diffusion_step=max_diffusion_step
        )
        self.candidate_act = DiffusionConvLayer(
            in_dim=cat_dim, h_dim=h_dim, n_adjs=n_adjs, max_diffusion_step=max_diffusion_step
        )

    def forward(self, x: Tensor, As: List[Tensor], h_0: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            x: input sequence
            As: list of adjacency matrices
            h_0: initial hidden state

        Returns:
            output: hidden state for each lookback time step
            h_n: last hidden state

        Shape:
            x: (B, L, N, C), where L denotes the input sequence length
            As: each A with shape (2, |E|), where |E| denotes the
                number edges
            h_0: (B, N, h_dim)
            output: (B, L, N, h_dim)
            h_n: (B, N, h_dim)
        """
        in_len = x.shape[1]

        output = []
        for t in range(in_len):
            x_t = x[:, t, ...]  # (B, N, C)
            if t == 0:
                h_t = None
                h_prev = self._init_hidden_state(x) if h_0 is None else h_0
            else:
                h_prev = h_t

            gate = F.sigmoid(self.gate(torch.cat([h_prev, x_t], dim=-1), As))
            r_t, u_t = torch.split(gate, self.h_dim, dim=-1)  # (B, N, h_dim)
            c_t = F.tanh(self.candidate_act(torch.cat([r_t * h_prev, x_t], dim=-1), As))
            h_t = u_t * h_prev + (1 - u_t) * c_t  # (B, N, h_dim)

            output.append(h_t.unsqueeze(dim=1))
        output = torch.cat(output, dim=1)  # (B, L, N, h_dim)
        h_n = h_t

        return output, h_n

    def _init_hidden_state(self, x: Tensor) -> Tensor:
        """Initialize the initial hidden state."""
        batch_size, _, n_series = x.shape[:-1]
        h_0 = torch.zeros(batch_size, n_series, self.h_dim, device=x.device)

        return h_0


class STConvBlock(nn.Module):
    """Spatial-temporal convolutional block of STGCN.

    Args:
        in_dim: input feature dimension
        h_dims: list of hidden dimension
        n_series: number of nodes
        kernel_size: kernel size
        cheb_k: order of Chebyshev Polynomials Approximation
        act: activation function
        dropout: dropout ratio
    """

    def __init__(
        self,
        in_dim: int,
        h_dims: List[List[int]],
        n_series: int,
        kernel_size: int,
        cheb_k: int,
        act: str,
        dropout: float
    ) -> None:
        super(STConvBlock, self).__init__()

        # Model blocks
        # Temporal convolution layer
        self.tcn1 = TemporalConvLayer(in_channels=in_dim, out_channels=h_dims[0], kernel_size=kernel_size, act=act)
        self.tcn2 = TemporalConvLayer(in_channels=h_dims[1], out_channels=h_dims[2], kernel_size=kernel_size, act=act)
        # Chebyshev Graph convolution layer
        self.gcn = ChebGraphConv(in_dim=h_dims[0], h_dim=h_dims[1], cheb_k=cheb_k)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.ln = nn.LayerNorm([n_series, h_dims[2]])

    def forward(self, x: Tensor, As: List[Tensor]) -> Tensor:
        """Forward pass.

        Args:
            x: input sequence
            As: list of adjacency matrices

        Returns:
            h: output node embedding

        Shape:
            x: (B, C, N, L), where L denotes the input sequence length
            As: each A with shape (N, N)
        """
        h = self.tcn1(x)
        h = self.gcn(h, As)
        h = self.relu(h)
        h = self.tcn2(h)
        h = self.ln(h.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)  # (B, L, N, C') -> (B, C', N, L)
        h = self.dropout(h)

        return h


class GWNetLayer(nn.Module):
    """Spatio-temporal layer of GWNet.

    One layer is constructed by a graph convolution layer and a gated
    temporal convolution layer.

    Args:
        in_dim: input feature dimension
        h_dim: hidden dimension
        kernel_size: kernel size
        dilation_factor: dilation factor
        gcn: if True, apply graph convolution
        n_adjs: number of adjacency matrices
            *Note: Bidirectional transition matrices are used in the
                original proposal.
        gcn_depth: depth of graph convolution
        gcn_dropout: droupout ratio in graph convolution layer
        bn: if True, apply batch normalization to output node embedding
    """

    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        kernel_size: int,
        dilation_factor: int,
        gcn: bool = True,
        n_adjs: int = 3,
        gcn_depth: int = 2,
        gcn_dropout: float = 0.3,
        bn: bool = True,
    ) -> None:
        super(GWNetLayer, self).__init__()

        # Netwrok parameters
        self.gcn = gcn

        # Model blocks
        # Gated temporal convolution layer
        self.tcn = GatedTCN(in_dim=in_dim, h_dim=h_dim, kernel_size=kernel_size, dilation_factor=dilation_factor)

        # Graph convolution layer
        if gcn:
            self.gcn = InfoPropLayer(in_dim=h_dim, h_dim=in_dim, n_adjs=n_adjs, depth=gcn_depth, dropout=gcn_dropout)
        else:
            self.resid = Linear2d(in_features=h_dim, out_features=h_dim)

        if bn:
            self.bn = nn.BatchNorm2d(in_dim)
        else:
            self.bn = None

    def forward(self, x: Tensor, As: Optional[List[Tensor]] = None) -> Tensor:
        """Forward pass.

        Args:
            x: input sequence
            As: list of adjacency matrices

        Returns:
            h_tcn: intermediate node embedding output by GatedTCN
            h: output node embedding

        Shape:
            x: (B, C, N, L), where L denotes the input sequence length
            As: each A with shape (N, N)
            h_tcn: (B, h_dim, N, L')
            h: (B, C, N, L')
        """
        x_resid = x

        # Gated temporal convolution layer
        h_tcn = self.tcn(x)

        # Graph convolution layer
        if self.gcn:
            h = self.gcn(h_tcn, As)
        else:
            h = self.resid(h_tcn)

        _, h_dim, _, out_len = h.shape
        h = h + x_resid[:, :h_dim, :, -out_len:]
        if self.bn is not None:
            h = self.bn(h)

        return h_tcn, h


class MTGNNLayer(TConvBaseModule):
    """Spatio-temporal layer of MTGNN.

    One layer is constructed by a mix-hop propagation layer and a
    dilated inception layer.

    Args:
        n_layers: number of stacked layers till the current layer
        n_series: number of series
        in_len: input sequence length
            *Note: This is the length of the raw input sequence, which
                might be padded to the receptive field.
        in_dim: input feature dimension
        h_dim: hidden dimension
        kernel_size: kernel size
        dilation_exponential: dilation exponential base
        tcn_dropout: droupout ratio in temporal convolution layer
        n_adjs: number of adjacency matrices
            *Note: Bidirectional transition matrices are used in the
                original proposal.
        gcn_depth: depth of graph convolution
        beta: retaining ratio for preserving locality
        ln_affine: if True, enable elementwise affine parameters
    """
    
    def __init__(
        self,
        n_layers: int,
        n_series: int,
        in_len: int,
        in_dim: int,
        h_dim: int,
        kernel_size: List[int],
        dilation_exponential: int,
        tcn_dropout: float = 0.3,
        gcn_depth: int = 2,
        beta: float = 0.05,
        ln_affine: bool = True,
    ) -> None:
        super(MTGNNLayer, self).__init__()

        # Netwrok parameters
        self._set_receptive_field(n_layers, dilation_exponential, kernel_size[-1])
        self.out_len = in_len - self.receptive_field + 1

        # Model blocks
        # Dilated inception layer
        self.tcn = GatedTCN(
            in_dim=in_dim,
            h_dim=h_dim,
            kernel_size=kernel_size,
            dilation_factor=dilation_exponential ** (n_layers - 1),
            dropout=tcn_dropout,
            conv_module=DilatedInception,
        )
        # Graph convolution layer
        self.gcn = InfoPropLayer(
            in_dim=h_dim,
            h_dim=in_dim,
            n_adjs=1,
            depth=gcn_depth,
            flow="tgt_to_src",  # Row norm
            normalize="asym",
            mix_prop=True,
            beta=beta,
        )
        self.gcn_t = InfoPropLayer(
            in_dim=h_dim,
            h_dim=in_dim,
            n_adjs=1,
            depth=gcn_depth,
            flow="tgt_to_src",  # Row norm
            normalize="asym",
            mix_prop=True,
            beta=beta,
        )
        # Layer normalization
        self.ln = nn.LayerNorm([in_dim, n_series, self.out_len], elementwise_affine=ln_affine)

    def forward(self, x: Tensor, As: List[Tensor]) -> Tensor:
        """Forward pass.

        Args:
            x: input sequence
            As: list of adjacency matrices

        Returns:
            h_tcn: intermediate node embedding output by dilated
                inception layer
            h: output node embedding

        Shape:
            x: (B, C, N, L), where L denotes the input sequence length
            As: each A with shape (N, N)
            h_tcn: (B, h_dim, N, L')
            h: (B, C, N, L')
        """
        x_resid = x

        # Dilated inception layer
        h_tcn = self.tcn(x)

        # Mix-hop propagation layer
        h = self.gcn(h_tcn, [As[0]]) + self.gcn_t(h_tcn, [As[1]])

        out_len = h.shape[-1]
        h = h + x_resid[..., -out_len:]
        h = self.ln(h)

        return h_tcn, h
    

class STSGCL(nn.Module):
    """Spatial-Temporal Synchronous Graph Convolutional Layer."""

    def __init__(
        self, 
        in_dim: int, 
        h_dim: int, 
        gcn_depth: int, 
        n_series: int,
        t_window: int,  
        act: str, 
        t_emb_dim: int, 
        s_emb_dim: int
    ) -> None:
        super(STSGCL, self).__init__()
        
        # Network parameters
        self.in_dim = in_dim
        self.n_series = n_series
        self.t_window = t_window

        # Model blocks
        self.stsgcm = nn.ModuleList()
        for _ in range(t_window - 2):
            self.stsgcm.append(STSGCM(in_dim=in_dim, h_dim=h_dim, gcn_depth=gcn_depth, n_series=n_series, act=act))

        self.spatial_emb, self.temporal_emb = None, None
        # Temporal embedding
        if t_emb_dim > 0:
            self.temporal_emb = nn.init.xavier_normal_(torch.empty(1, t_window, 1, t_emb_dim), gain=0.0003)
        # Spatial embedding
        if s_emb_dim > 0:
            self.spatial_emb = nn.init.xavier_normal_(torch.empty(1, 1, n_series, s_emb_dim), gain=0.0003)

    def forward(self, x: Tensor, A: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: input sequence
            A: adjacency matrix

        Shape:
            x: (B, L, N, C)
            A: (3N, 3N)
            output: (B, L - 2, N, C')
        """
        # Spatial Temporal embedding
        if self.spatial_emb is not None:
            x = x + self.spatial_emb.to(x.device)
        if self.temporal_emb is not None:
            x = x + self.temporal_emb.to(x.device)

        # Spatial-Temporal Synchronous Graph Convolutional Module
        x_convs = []
        for i in range(self.t_window - 2):
            h = x[:, i : i + 3, :, :]                               # (B, 3, N, C)
            h = h.reshape([-1, 3 * self.n_series, self.in_dim])     # (B, 3N, C)
            h = h.permute(1, 0, 2)                                  # (3N, B, C)
            h = self.stsgcm[i](h, A).permute(1, 0, 2)               # (B, N, C')
            x_convs.append(h)
        output = torch.stack(x_convs, dim=1)             # (B, T-2, N, C')

        return output


class AGCGRU(nn.Module):
    """Adaptive graph convolutional gated recurrent unit.

    Args:
        in_dim: input feature dimension
        h_dim: hidden state dimension
        emb_dim: embedding dimension
        cheb_k: order of chebyshev polynomial expansion
    """

    def __init__(self, in_dim: int, h_dim: int, emb_dim: int, cheb_k: int) -> None:
        super(AGCGRU, self).__init__()

        # Network parameters
        self.h_dim = h_dim

        # Model blocks
        cat_dim = in_dim + h_dim
        self.gate = NAPLConvLayer(in_dim=cat_dim, h_dim=h_dim * 2, emb_dim=emb_dim, cheb_k=cheb_k)
        self.candidate_act = NAPLConvLayer(in_dim=cat_dim, h_dim=h_dim, emb_dim=emb_dim, cheb_k=cheb_k)

    def forward(
        self, x: Tensor, node_embs: Tensor, As: Tensor, h_0: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            x: input sequence
            node_embs: node embeddings
            h_0: initial hidden state

        Returns:
            output: hidden state for each lookback time step
            h_n: last hidden state

        Shape:
            x: (B, L, N, C), where L denotes the input sequence length
            h_0: (B, N, h_dim)
            output: (B, L, N, h_dim)
            h_n: (B, N, h_dim)
        """
        in_len = x.shape[1]

        output = []
        for t in range(in_len):
            x_t = x[:, t, ...]  # (B, N, C)
            if t == 0:
                h_t = None
                h_prev = self._init_hidden_state(x) if h_0 is None else h_0
            else:
                h_prev = h_t

            gate = F.sigmoid(self.gate(torch.cat([h_prev, x_t], dim=-1), node_embs, As))
            r_t, u_t = torch.split(gate, self.h_dim, dim=-1)  # (B, N, h_dim)
            c_t = F.tanh(self.candidate_act(torch.cat([r_t * h_prev, x_t], dim=-1), node_embs, As))
            h_t = u_t * h_prev + (1 - u_t) * c_t  # (B, N, h_dim)

            output.append(h_t.unsqueeze(dim=1))
        output = torch.cat(output, dim=1)  # (B, L, N, h_dim)
        h_n = h_t

        return output, h_n

    def _init_hidden_state(self, x: Tensor) -> Tensor:
        """Initialize the initial hidden state."""
        batch_size, _, n_series = x.shape[:-1]
        h_0 = torch.zeros(batch_size, n_series, self.h_dim, device=x.device)

        return h_0


class GMANAttentionBlock(nn.Module):
    """Spatial-temporal attention block."""

    def __init__(
        self, in_dim: int, h_dim: int, n_heads: int, mask: bool = True, act: str = "relu", bn: bool = True
    ) -> None:
        super(GMANAttentionBlock, self).__init__()

        # Model blocks
        # Spatial attention
        self.spatial_attn = AttentionLayer(
            in_dim=in_dim * 2, h_dim=h_dim, n_heads=n_heads, act=act, bn=bn, reset_params=True
        )
        # Temporal attention
        self.temporal_attn = AttentionLayer(
            in_dim=in_dim * 2, h_dim=h_dim, n_heads=n_heads, mask=mask, act=act, bn=bn, reset_params=True
        )
        # Gated Fusion
        self.gated_fusion = GatedFusion(h_dim=h_dim)

    def forward(self, x: Tensor, st_emb: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: input sequence
            st_emb: Spatial-Temporal embedding

        Returns:
            output: output hidden state
        
        Shape:
            x: (B, L, N, in_dim)
            st_emb: (B, L, N, in_dim)
            output: (B, L, N, h_dim)
        """
        h = torch.cat((x, st_emb), dim=-1)                  # Concat along channel, (B, L, N, 2 * in_dim)
        # Spatial attention
        hs = self.spatial_attn(h, h, h)                     # (B, L, N, h_dim)
        # Temporal attention
        h = h.transpose(1, 2)                               # (B, N, L, h_dim)
        ht = self.temporal_attn(h, h, h).transpose(1, 2)    # (B, L, N, h_dim)
        # Gated Fusion
        h = self.gated_fusion(hs, ht)              # (B, L, N, h_dim)
        output = x + h                             # (B, L, N, h_dim)

        return output


class SCINetTree(nn.Module):
    def __init__(
        self, in_dim: int, h_ratio: int, kernel_size: int, groups: int, dropout: float, INN: bool, current_level: int
    ) -> None:
        """SCINet Tree.

        Args:
            in_dim: input dimension
            h_ratio: in_dim * h_ratio = hidden dimension
            kernel_size: kernel size
            groups: groups
            dropout: dropout ratio
            INN: if True, apply interactive learning
            current_level: current level of tree
        """

        super(SCINetTree, self).__init__()

        # Network parameters
        self.current_level = current_level

        # Model blocks
        self.sciblock = SCIBlock(
            in_dim=in_dim,
            h_ratio=h_ratio,
            kernel_size=kernel_size,
            groups=groups,
            dropout=dropout,
            INN=INN
        )

        if current_level != 0:
            self.tree_odd = SCINetTree(
                in_dim=in_dim,
                h_ratio=h_ratio,
                kernel_size=kernel_size,
                groups=groups,
                dropout=dropout,
                INN=INN,
                current_level=current_level - 1
            )
            self.tree_even = SCINetTree(
                in_dim=in_dim,
                h_ratio=h_ratio,
                kernel_size=kernel_size,
                groups=groups,
                dropout=dropout,
                INN=INN,
                current_level=current_level - 1
            )
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: input sequence
        """
        h_even, h_odd = self.sciblock(x)

        if self.current_level == 0:
            return self.concat_and_realign(h_even, h_odd)
        else:
            return self.concat_and_realign(self.tree_even(h_even), self.tree_odd(h_odd))
    
    def concat_and_realign(self, x_even: Tensor, x_odd: Tensor) -> Tensor:
        """Concat & Realign.

        Args:
            x_even: even sub-sequence
            x_odd: odd sub-sequence

        Shape:
            x_even: (B, L, D)
            x_odd: (B, L, D)
            output: (B, L', D)
        """

        x_even = x_even.permute(1, 0, 2)    # (L, B, D)
        x_odd = x_odd.permute(1, 0, 2)      # (L, B, D)

        even_len = x_even.shape[0]
        odd_len = x_odd.shape[0]
        min_len = min((odd_len, even_len))

        output = []
        for i in range(min_len):
            output.append(x_even[i].unsqueeze(0))
            output.append(x_odd[i].unsqueeze(0))
        
        if odd_len < even_len: 
            output.append(x_even[-1].unsqueeze(0))

        output = torch.cat(output, dim=0).permute(1, 0, 2)

        return  output