"""
Common layers in model architecture.
Author: JiaWei Jiang

This file contains commonly used nn layers in diversified model arch.
"""
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# from torch_geometric.nn.inits import zeros


class Swish(nn.Module):
    """Activation function, swish."""

    def __init__(self) -> None:
        self.name = self.__class__.__name__
        super(Swish, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        """Forward passing.

        Parameters:
            x: input variables

        Return:
            x: non-linearly transformed variables
        """
        x = x * torch.sigmoid(x)

        return x


class Mish(nn.Module):
    """Activation function, mish."""

    def __init__(self) -> None:
        self.name = self.__class__.__name__
        super(Mish, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        """Forward passing.

        Parameters:
            x: input variables

        Return:
            x: non-linearly transformed variables
        """
        x = x * F.tanh(F.softplus(x))

        return x


class GLU(nn.Module):
    """Gated linear unit.

    Parameters:
        in_dim: input feature dimension
        h_dim: hidden dimension of linear layer
        dropout: dropout ratio
    """

    def __init__(self, in_dim: int, h_dim: int, dropout: Optional[float] = None):
        self.name = self.__class__.__name__
        super(GLU, self).__init__()

        # Network parameters
        self.in_dim = in_dim
        self.h_dim = h_dim

        # Model blocks
        self.l1 = nn.Linear(in_dim, h_dim)
        self.l2 = nn.Linear(in_dim, h_dim)
        self.act = nn.Sigmoid()
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """
        Shape:
            x: (*, in_dim)
            output: (*, h_dim)
        """
        if self.dropout is not None:
            x = self.dropout(x)

        # Linear transformation
        x_lin = self.l1(x)

        # Gating mechanism
        x_gate = self.act(self.l2(x))

        # Output
        output = x_gate * x_lin

        return output


class SwapNoiseAdder(nn.Module):
    """Add noise to generate corrupted features based on swapping
    mechanism.

    [ ] Swap based on some prior knowledge (e.g., node distance)

    Parameters:
        doping_ratio: ratio of features to dope
        doping_scale: range to swap feature values, either global or
            in-batch
    """

    def __init__(self, doping_ratio: float = 0.15, doping_scale: str = "global"):
        self.name = self.__class__.__name__
        super(SwapNoiseAdder, self).__init__()

        self.doping_ratio = doping_ratio
        self.doping_scale = doping_scale

    def forward(self, x: Tensor) -> Tensor:
        batch_size, n_series, t_window = x.shape

        if self.doping_scale == "global":
            x = x.contiguous().view(batch_size * n_series, -1)

        doping_mask = torch.bernoulli(torch.full(x.shape, self.doping_ratio, device=x.device))

        if self.doping_scale == "global":
            x_corrupted = torch.where(doping_mask == 1, x[torch.randperm(x.size(0))], x)
            x_corrupted = x_corrupted.contiguous().view(batch_size, n_series, -1)
        else:
            x_corrupted = torch.where(doping_mask == 1, x[:, torch.randperm(x.size(1)), :], x)

        return x_corrupted


class MultiInputSequential(nn.Sequential):
    """Sequential module supporting multiple inputs."""

    def forward(self, *inputs: Any, **kwargs: Any) -> Tensor:
        for module in self._modules.values():
            # Pass through all modules sequentially
            # Note that inputs can be the outputs of the previous layer
            if type(inputs) == tuple:
                inputs = module(*inputs, **kwargs)
            else:
                inputs = module(inputs, **kwargs)

        return inputs


class MixProp(nn.Module):
    """Dynamic mix-hop propagation layer.

    The implementation follows the MTGNN official release.

    Parameters:
        c_in: input channel number
        c_out: output channel number
        gcn_depth: depth of graph convolution
        alpha: retaining ratio of the original state of node features
        dropout: dropout ratio
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        gcn_depth: int,
        alpha: float = 0.05,
        dropout: Optional[float] = None,
    ):
        self.name = self.__class__.__name__
        super(MixProp, self).__init__()

        # Network parameters
        self.c_in = c_in
        self.c_out = c_out
        self.gcn_depth = gcn_depth
        self.alpha = alpha

        # Model blocks
        self.l_slc = nn.Linear((gcn_depth + 1) * c_in, c_out)
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, x: Tensor, A: Tensor, hs: Optional[List[Tensor]] = None) -> Tuple[Tensor, None]:
        """Forward pass.

        Parameters:
            x: node features
            A: adjacency matrix
            hs: always ignored

        Return:
            h: final node embedding

        Shape:
            x: (B, N, emb_dim) or (B, N, *)
                *Note: * can be 1- (feature vec) or 2-D (feature map)
            A: (B, N, N), B is always equal to 1 for static GS
            h: (B, N, *)
        """
        assert x.dim() == 3, "Shape of node features doesn't match (B, N, C)."

        # Information propagation
        h_0 = x  # (B, N, *)
        h = x
        for hop in range(self.gcn_depth):
            if x.dim() == 3:
                # Node feature vector
                h = self.alpha * x + (1 - self.alpha) * torch.einsum(
                    "bwc,bvw->bvc", (h, A)  # (B, N, *), (B, N, N)
                )  # (B, N, *)
            elif x.dim() == 4:
                # Node feature map
                pass
            h_0 = torch.cat((h_0, h), dim=-1)  # Concat along channel

        # Information selection
        h = self.l_slc(h_0)

        return h, None


class HopAwareRecGConv(nn.Module):
    """Hop-aware rectified graph convolution module.

    Parameters:
        c_in: input channel number
        c_out: output channel number
        gcn_depth: depth of graph convolution
        alpha: retaining ratio of the original state of node features
        dropout: dropout ratio
        hop_aware_rectify_fn: hop-aware rectifying function
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        gcn_depth: int,
        alpha: float = 0.05,
        dropout: Optional[float] = None,
        hop_aware_rectify_fn: str = "mean",
    ):
        self.name = self.__class__.__name__
        super(HopAwareRecGConv, self).__init__()

        # Network parameters
        self.c_in = c_in
        self.c_out = c_out
        self.gcn_depth = gcn_depth
        self.alpha = alpha
        self.hop_aware_rectify_fn = hop_aware_rectify_fn

        # Model blocks
        # Hop-aware rectifiers
        self.hop_aware_rectifiers = nn.ModuleList()
        for hop in range(gcn_depth):
            if hop_aware_rectify_fn == "linear":
                self.hop_aware_rectifiers.append(nn.Linear(c_in * 2, c_in))
            elif hop_aware_rectify_fn == "gru":
                self.hop_aware_rectifiers.append(nn.GRU(c_in, c_in, batch_first=True, dropout=0))
            elif hop_aware_rectify_fn == "glu":
                self.hop_aware_rectifiers.append(GLU(c_in, c_in, dropout=0.3))
        # Dropout
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        # Information selection
        self.l_slc = nn.utils.weight_norm(nn.Linear((gcn_depth + 1) * c_in, c_out), dim=None)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.l_slc.weight)

    def forward(self, x: Tensor, A: Tensor, hs: Optional[List[Tensor]] = None) -> Tuple[Tensor, List[Tensor]]:
        """Forward pass.

        Parameters:
            x: node features
            A: adjacency matrix
            hs: hop-aware intermediate node embedding over A from layer
                (l-1)

        Return:
            h: final node embedding
            h_latent: hop-aware intermediate node embedding over A for
                the next `_HARDPurGLayer`

        Shape:
            x: (B, N, C'), (B, N, d_conv) or (B, N, *)
                *Note: * can be 1- (feature vec) or 2-D (feature map)
            A: (B, N, N), B is always equal to 1 for SGS
            h: (B, N, d_conv)
        """
        assert x.dim() == 3, "Shape of node features is wrong."
        batch_size, n_series, n_feats = x.shape

        # Hop-aware rectified graph convolution
        h_latent = []
        h_mix = x
        h = x
        for hop in range(self.gcn_depth):
            # Message passing and aggregation
            h = self.alpha * x + (1 - self.alpha) * torch.einsum(
                "bwc,bvw->bvc", (h, A)  # (B, N, C'), (B, N, N)
            )  # (B, N, C')

            if self.dropout is not None:
                h = self.dropout(h)

            # Hop-aware rectifying
            if self.hop_aware_rectify_fn == "linear":
                if hs is not None:
                    h = self.hop_aware_rectifiers[hop](torch.cat([h, hs[hop]], dim=-1))
            elif self.hop_aware_rectify_fn == "gru":
                h = h.contiguous().view(batch_size * n_series, 1, -1)  # (B * N, L, C'), L = 1
                if hs is None:
                    # Initial hidden states are zeros for the first layer
                    _, h = self.hop_aware_rectifiers[hop](h)  # (1, B * N, C')
                else:
                    _, h = self.hop_aware_rectifiers[hop](h, hs[hop])  # (1, B * N, C')
            elif self.hop_aware_rectify_fn == "glu":
                if hs is not None:
                    h = h + self.hop_aware_rectifiers[hop](hs[hop])
            elif self.hop_aware_rectify_fn == "mean":
                if hs is not None:
                    h = h + hs[hop]  # / 2

            h_latent.append(h)  # Fed into the next HARDPurG layer

            h = h.contiguous().view(batch_size, n_series, -1)  # (B, N, C')
            h_mix = torch.cat((h_mix, h), dim=-1)  # Concat along channel

        # Information selection
        h = self.l_slc(h_mix)

        return h, h_latent


class AuxInfoAdder(nn.Module):
    """Add auxiliary information along feature (i.e., channel) axis.

    It's used in traffic forecasting scenarios, where time identifiers
    are considered to be auxiliary information.
    """

    N_TIMES_IN_DAY: int = 288
    N_DAYS_IN_WEEK: int = 7

    def __init__(self) -> None:
        super(AuxInfoAdder, self).__init__()

        self.lookback_idx = torch.arange(12)

    def forward(self, x: Tensor, tid: Optional[Tensor] = None, diw: Optional[Tensor] = None) -> Tensor:
        """Forward pass.

        Shape:
            x: (B, N, T) or
            tid: (B, )
            diw: (B, )
        """
        batch_size, n_series, t_window = x.shape
        x = torch.unsqueeze(x, dim=1)  # (B, C, N, T)

        if self.lookback_idx.device != x.device:
            self.lookback_idx = self.lookback_idx.to(x.device)

        if tid is not None:
            tid_expand = tid.unsqueeze(dim=-1).unsqueeze(dim=-1).expand(-1, n_series, t_window)  # (B, N, T)
            tid_back = (((tid_expand + self.N_TIMES_IN_DAY) - self.lookback_idx) % self.N_TIMES_IN_DAY).flip(
                dims=[2]
            ) / self.N_TIMES_IN_DAY
            x = torch.cat([x, tid_back.unsqueeze(dim=1)], dim=1)
        if diw is not None:
            diw_expand = diw.unsqueeze(dim=-1).unsqueeze(dim=-1).expand(-1, n_series, t_window)  # (B, N, T)
            cross_day_mask = ((tid_expand - self.lookback_idx).flip(dims=[2]) < 0).int()
            diw_back = (
                ((diw_expand + self.N_DAYS_IN_WEEK) - cross_day_mask) % self.N_DAYS_IN_WEEK / self.N_DAYS_IN_WEEK
            )
            x = torch.cat([x, diw_back.unsqueeze(dim=1)], dim=1)

        return x
