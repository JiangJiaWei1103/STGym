"""
Sub-layers.
Author: JiaWei Jiang
"""
from typing import List, Optional, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# Spatial pattern extractor
class DiffusionConvLayer(nn.Module):
    """Diffusion convolutional layer."""

    def __init__(
        self, in_dim: int, h_dim: int, n_adjs: int = 2, max_diffusion_step: int = 2, act: Optional[str] = None
    ) -> None:
        super(DiffusionConvLayer, self).__init__()

        # Network parameters
        self.in_dim = in_dim
        self.max_diffusion_step = max_diffusion_step
        self.n_node_embs = 1 + n_adjs * max_diffusion_step  # Number of node embeddings (i.e., feature matrices)

        # Model blocks
        self.conv_filter = nn.Linear(in_dim * self.n_node_embs, h_dim)
        if act is not None:
            if act == "relu":
                self.act = nn.ReLU()
            elif act == "sigmoid":
                self.act = nn.Sigmoid()
        else:
            self.act = None

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_normal_(self.conv_filter.weight.data)
        nn.init.constant_(self.conv_filter.bias.data, val=0.0)

    def forward(self, x: Tensor, As: List[Tensor]) -> Tensor:
        """Forward pass.

        Parameters:
            x: input node features
            As: list of adjacency matrices

        Return:
            h: diffusion convolution output

        Shape:
            x: (B, N, C)
            As: each A with shape (2, |E|), where |E| denotes the
                number edges
            h: (B, N, h_dim)
        """
        batch_size, n_series, _ = x.shape

        x = x.permute(1, 2, 0)  # (N, C, B)
        x = x.reshape(n_series, -1)  # (N, C * B)
        x_convs = x.unsqueeze(0)  # (1, N, C * B)
        for A in As:
            for k in range(1, self.max_diffusion_step + 1):
                if k == 1:
                    x_conv = torch.mm(A, x)
                else:
                    x_conv = 2 * torch.mm(A, x_conv) - x
                    x = x_conv
                x_convs = torch.cat([x_convs, x_conv.unsqueeze(0)], dim=0)

        x = x_convs.reshape(self.n_node_embs, n_series, self.in_dim, batch_size)
        x = x.permute(3, 1, 2, 0)  # (B, N, C, n_node_embs)
        x = x.reshape(batch_size * n_series, -1)
        h = self.conv_filter(x).reshape(batch_size, n_series, -1)  # (B, N, h_dim)

        return h


class GCN2d(nn.Module):
    """Graph convolution layer over 2D planes.

    GCN2d applies graph convolution over graph signal represented by
    2D node embedding planes.

    Parameters:
        flow: flow direction of the message passing, the choices are
            {"src_to_tgt", "tgt_to_src"}
        mix_prop: if True, mix-hop propagation is activated
    """

    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        n_adjs: int,
        depth: int,
        flow: str = "src_to_tgt",
        normalize: bool = False,
        mix_prop: bool = False,
        beta: Optional[float] = None,
        dropout: Optional[float] = None,
    ) -> None:
        super(GCN2d, self).__init__()

        # Network parameters
        self.depth = depth
        self.flow = flow
        self.normalize = normalize
        self.mix_prop = mix_prop
        if mix_prop:
            assert beta is not None, "Please specify retraining ratio for preserving locality."
        self.beta = beta
        self.n_node_embs = 1 + n_adjs * depth  # Number of node embeddings (i.e., feature matrices)
        if flow == "src_to_tgt":
            self._flow_eq = "bcvl,vw->bcwl"
        else:
            self._flow_eq = "bcwl,vw->bcvl"

        # Model blocks
        self.conv_filter = Linear2d(in_dim * self.n_node_embs, h_dim)
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x: Tensor, As: List[Tensor]) -> Tensor:
        """Forward pass.

        Parametersi:
            x: input node features
            As: list of adjacency matrices

        Return:
            h: graph convolution output

        Shape:
            x: (B, in_dim, N, L)
            As: each A with shape (N, N)
            h: (B, h_dim, N, L)
        """
        # Information propagation
        x_convs = [x]  # k == 0
        h_in = x if self.mix_prop else None
        for A in As:
            if self.normalize:
                A = self._normalize(A)
            for k in range(1, self.depth + 1):
                if k == 1:
                    x_conv = x

                if self.mix_prop:
                    x_conv = torch.einsum(self._flow_eq, (x_conv, A))
                else:
                    x_conv = self.beta * h_in + (1 - self.beta) * torch.einsum(self._flow_eq, (x_conv, A))
                x_convs.append(x_conv)

        h = torch.cat(x_convs, dim=1)
        h = self.conv_filter(h)
        if self.dropout is not None:
            h = self.dropout(h)

        return h

    def _normalize(self, A: Tensor) -> Tensor:
        """Normalize adjacency matrix."""
        A = A + torch.eye(A.size(0)).to(A.device)
        A_to_norm = A.T if self.flow == "src_to_tgt" else A
        D = torch.sum(A_to_norm, dim=1).reshape(-1, 1)
        A = A_to_norm / D
        if self.flow == "src_to_tgt":
            A = A.T

        return A


# Temporal pattern extractor
class GatedTCN(nn.Module):
    """Gated temporal convolution layer.

    Parameters:
        conv_module: customized convolution module
    """

    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        kernel_size: Union[int, List[int]],
        dilation_factor: int,
        dropout: Optional[float] = None,
        conv_module: Optional[Type[nn.Module]] = None,
    ) -> None:
        super(GatedTCN, self).__init__()

        # Model blocks
        if conv_module is None:
            self.filter = nn.Conv2d(
                in_channels=in_dim, out_channels=h_dim, kernel_size=(1, kernel_size), dilation=dilation_factor
            )
            self.gate = nn.Conv2d(
                in_channels=in_dim, out_channels=h_dim, kernel_size=(1, kernel_size), dilation=dilation_factor
            )
        else:
            self.filter = conv_module(
                in_channels=in_dim, out_channels=h_dim, kernel_size=kernel_size, dilation=dilation_factor
            )
            self.gate = conv_module(
                in_channels=in_dim, out_channels=h_dim, kernel_size=kernel_size, dilation=dilation_factor
            )

        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters:
            x: input sequence

        Return:
            h: output sequence

        Shape:
            x: (B, C, N, L), where L denotes the input sequence length
            h: (B, h_dim, N, L')
        """
        x_filter = F.tanh(self.filter(x))
        x_gate = F.sigmoid(self.gate(x))
        h = x_filter * x_gate
        if self.dropout is not None:
            h = self.dropout(h)

        return h


class DilatedInception(nn.Module):
    """Dilated inception layer.

    Note that `out_channels` will be split across #kernels.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: List[int], dilation: int) -> None:
        super(DilatedInception, self).__init__()

        # Network parameters
        n_kernels = len(kernel_size)
        assert out_channels % n_kernels == 0, "out_channels must be divisible by #kernels."
        h_dim = out_channels // n_kernels

        # Model blocks
        self.convs = nn.ModuleList()
        for k in kernel_size:
            self.convs.append(
                nn.Conv2d(in_channels=in_channels, out_channels=h_dim, kernel_size=(1, k), dilation=dilation)
            )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters:
            x: input sequence

        Return:
            h: output sequence

        Shape:
            x: (B, C, N, L)
            h: (B, out_channels, N, L')
        """
        x_convs = []
        for conv in self.convs:
            x_conv = conv(x)
            x_convs.append(x_conv)

        # Truncate according to the largest filter
        out_len = x_convs[-1].shape[-1]
        for i, x_conv in enumerate(x_convs):
            x_convs[i] = x_conv[..., -out_len:]
        h = torch.cat(x_convs, dim=1)

        return h


# Common
class Linear2d(nn.Module):
    """Linear layer over 2D plane.

    Linear2d applies linear transformation along channel dimension of
    2D planes.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear2d, self).__init__()

        # Model blocks
        self.lin = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=(1, 1), bias=bias)

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

        return output
