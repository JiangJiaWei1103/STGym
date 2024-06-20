"""
Temporal-layers.
Author: JiaWei Jiang, ChunWei Shen
"""
from typing import List, Optional, Type, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .common_layers import Align

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