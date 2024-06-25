"""
Temporal-layers.
Author: JiaWei Jiang, ChunWei Shen
"""
from typing import List, Optional, Type, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .common_layers import Align, Split

# Temporal pattern extractor
class GatedTCN(nn.Module):
    """Gated temporal convolution layer.

    Args:
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

        Args:
            x: input sequence

        Returns:
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
                nn.Conv2d(in_channels=in_channels, out_channels=h_dim, kernel_size=(1, k), dilation=(1, dilation))
            )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: input sequence

        Returns:
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
    

class TemporalConvLayer(nn.Module):
    """Temporal Convolution Layer."""

    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        act: str = None
    ):
        super(TemporalConvLayer, self).__init__()

        # Network parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.act = act

        # Model blocks
        self.align = Align(in_channels, out_channels)

        if act == "glu" or act == "gtu":
            self.causal_conv = nn.Conv2d(
                in_channels=in_channels, 
                out_channels=2 * out_channels, 
                kernel_size=(1, kernel_size)
            )
        else:
            self.causal_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, kernel_size)
            )
            
        if act is not None:
            if act == "relu":
                self.act_func = nn.ReLU()
            elif act == "leakyrelu":
                self.act_func = nn.LeakyReLU()
            elif act == "tanh":
                self.act_func = nn.Tanh()
            elif act == "sigmoid":
                self.act_func = nn.Sigmoid()
            elif act == "silu":
                self.act_func = nn.SiLU()
        else:
            self.act_func = None

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: input sequence

        Returns:
            h: output sequence

        Shape:
            x: (B, C, N, L)
            h: (B, out_channels, N, L')
        """
        x_resid = self.align(x)[..., (self.kernel_size - 1):]

        x_causal = self.causal_conv(x)
        if self.act == "glu" or self.act == "gtu":
            x_p = x_causal[:, :self.out_channels, :, :]
            x_q = x_causal[:, -self.out_channels:, :, :]
            if self.act == "glu": 
                h = torch.mul((x_p + x_resid), torch.sigmoid(x_q))
            else:
                h = torch.mul(torch.tanh(x_p + x_resid), torch.sigmoid(x_q))
        elif self.act_func is not None:
            h = self.act_func(x_causal + x_resid)
        
        return h


class SCIBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        h_ratio: int = 1,
        kernel_size: int = 5,
        groups: int = 1,
        dropout: float = 0.5,
        split: bool = True,
        INN: bool = True
    ) -> None:
        """SCI-Block.

        Args:
            in_dim: input dimension
            h_ratio: in_dim * h_ratio = hidden dimension
            kernel_size: kernel size
            groups: groups
            dropout: dropout ratio
            split: if True, apply split
            INN: if True, apply interactive learning
        """

        super(SCIBlock, self).__init__()

        # Network parameters
        self.splitting = split
        self.INN = INN
        # size of the padding
        if kernel_size % 2 == 0:
            pad_l = (kernel_size - 2) // 2 + 1
            pad_r = (kernel_size) // 2 + 1
        else:
            pad_l = (kernel_size - 1) // 2 + 1
            pad_r = (kernel_size - 1) // 2 + 1

        # Model blocks
        self.split = Split()
        # Convolutional module
        self.conv_phi = SCIConv(
            in_dim=in_dim,
            h_ratio=h_ratio,
            kernel_size=kernel_size,
            padding=(pad_l, pad_r),
            groups=groups,
            dropout=dropout
        )
        self.conv_psi = SCIConv(
            in_dim=in_dim,
            h_ratio=h_ratio,
            kernel_size=kernel_size,
            padding=(pad_l, pad_r),
            groups=groups,
            dropout=dropout
        )
        self.conv_p = SCIConv(
            in_dim=in_dim,
            h_ratio=h_ratio,
            kernel_size=kernel_size,
            padding=(pad_l, pad_r),
            groups=groups,
            dropout=dropout
        )
        self.conv_u = SCIConv(
            in_dim=in_dim,
            h_ratio=h_ratio,
            kernel_size=kernel_size,
            padding=(pad_l, pad_r),
            groups=groups,
            dropout=dropout
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            x: input sequence
        
        Shape:
            x: (B, L, N)
            h_even: (B, L', N)
            h_odd: (B, L', N)
        """
        # Split
        if self.split:
            x_even, x_odd = self.split(x)
        else:
            x_even, x_odd = x

        # Interactive learning
        if self.INN:
            x_even = x_even.permute(0, 2, 1)    # (B, N, L)
            x_odd = x_odd.permute(0, 2, 1)      # (B, N, L)

            xs_odd = x_odd.mul(torch.exp(self.conv_phi(x_even)))
            xs_even = x_even.mul(torch.exp(self.conv_psi(x_odd)))

            h_even = xs_even + self.conv_u(xs_odd)
            h_odd = xs_odd - self.conv_p(xs_even)

            h_even = h_even.permute(0, 2, 1)    # (B, L', N)
            h_odd = h_odd.permute(0, 2, 1)      # (B, L', N)

            return h_even, h_odd
        else:
            x_even = x_even.permute(0, 2, 1)    # (B, N, L)
            x_odd = x_odd.permute(0, 2, 1)      # (B, N, L)

            h_even = x_even + self.conv_u(x_odd)
            h_odd = x_odd - self.conv_p(x_even)
        
            h_even = h_even.permute(0, 2, 1)   # (B, L', N)
            h_odd = h_odd.permute(0, 2, 1)     # (B, L', N)

            return h_even, h_odd


class SCIConv(nn.Module):
    """SCINet 1d convolutional module."""

    def __init__(
        self, 
        in_dim: int, 
        h_ratio: float, 
        kernel_size: int,
        padding: Tuple[int], 
        groups: int, 
        dropout: float
    ) -> None:
        super(SCIConv, self).__init__()

        # Model blocks
        self.conv = nn.Sequential(
            nn.ReplicationPad1d(padding=padding),
            nn.Conv1d(
                in_channels=in_dim,
                out_channels=int(in_dim * h_ratio),
                kernel_size=kernel_size,
                groups=groups
            ),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(
                in_channels=int(in_dim * h_ratio),
                out_channels=in_dim,
                kernel_size=3,
                groups=groups
            ),
            nn.Tanh()
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: input sequence

        Returns:
            output: output sequence

        Shape:
            x: (B, N, L)
            output: (B, N, L')
        """
        output = self.conv(x)

        return output