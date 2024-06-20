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
    def __init__(self, in_features: int, out_features: int, bias: bool = True,) -> None:
        super(Linear2d, self).__init__()

        # Model blocks
        self.lin = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=(1, 1), bias=bias)

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