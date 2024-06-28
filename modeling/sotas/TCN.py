"""
Baseline method, TCN.
Author: ChunWei Shen
"""
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from modeling.module.tconv import TConvBaseModule
from modeling.module.common_layers import Linear2d

class TCN(TConvBaseModule):
    """TCN framework.

        Args:
            in_dim: input feature dimension
            skip_dim: output dimension of skip connection
            end_dim: hidden dimension of output layer
            in_eln: input sequence length
            out_len: output sequence length
            n_layers: number of TCN layers
            tcn_h_dim: hidden dimension of TCN
            kernel_size: kernel size
            dilation_exponential: dilation exponential base
            dropout: dropout ratio
    """
    
    def __init__(
        self,
        in_dim: int,
        skip_dim: int,
        end_dim: int,
        in_len: int,
        out_len: int,
        st_params: Dict[str, Any]
    ) -> None:
        self.name = self.__class__.__name__
        super(TCN, self).__init__()

        # Network parameters
        self.st_params = st_params
        n_layers = st_params["n_layers"]
        tcn_h_dim = st_params["tcn_h_dim"]
        kernel_size = st_params["kernel_size"]
        dilation_exponential = st_params["dilation_exponential"]
        dropout = st_params["dropout"]
        self.in_len = in_len
        self.out_len = out_len
        # Receptive field and sequence length
        self._set_receptive_field(n_layers, dilation_exponential, kernel_size)
        if in_len < self.receptive_field:
            in_len = self.receptive_field

        # Model blocks
        # Input linear layer
        self.in_lin = Linear2d(in_dim, tcn_h_dim)
        # TCN
        self.in_skip = nn.Sequential(
            nn.Dropout(dropout), nn.Conv2d(in_channels=in_dim, out_channels=skip_dim, kernel_size=(1, in_len))
        )
        self.tcn_layers = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        for layer in range(n_layers):
            self.tcn_layers.append(
                TCNLayer(
                    n_layers=layer + 1,
                    in_len=in_len,
                    in_dim=tcn_h_dim,
                    h_dim=tcn_h_dim,
                    kernel_size=kernel_size,
                    dilation_exponential=dilation_exponential
                )
            )
            layer_out_len = self.tcn_layers[layer].out_len
            self.skip_convs.append(
                nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Conv2d(in_channels=tcn_h_dim, out_channels=skip_dim, kernel_size=(1, layer_out_len)),
                )
            )
        # Output layer
        self.output = nn.Sequential(nn.ReLU(), Linear2d(skip_dim, end_dim), nn.ReLU(), Linear2d(end_dim, out_len))

    def forward(self, x: Tensor, As: List[Tensor], **kwargs: Any) -> Tuple[Tensor, None, None]:
        """Forward pass.

        Parameters:
            x: input sequence

        Return:
            output: prediction

        Shape:
            x: (B, P, N, C)
            output: (B, Q, N)
        """
        # Input linear layer
        x = x.permute(0, 3, 2, 1)  # (B, C, N, P)
        if self.in_len < self.receptive_field:
            x = self._pad_seq_to_rf(x)

        # Stacked spatio-temporal layers
        x_skip = self.in_skip(x)
        h = self.in_lin(x)  # (B, tcn_in_dm, N, P)
        for layer, tcn_layer in enumerate(self.tcn_layers):
            # Capture inflow and outflow patterns
            h = tcn_layer(h)
            # Skip-connect to the output module
            x_skip = x_skip + self.skip_convs[layer](h)  # (B, skip_dim, N, 1)

        # Output layer
        output = self.output(x_skip).squeeze(dim=-1)  # (B, Q, N)

        return output, None, None
    

class TCNLayer(TConvBaseModule):
    def __init__(
        self, n_layers: int, in_len: int, in_dim: int, h_dim: int, kernel_size: int, dilation_exponential: int
    ) -> None:
        super(TCNLayer, self).__init__()

        # Netwrok parameters
        self._set_receptive_field(n_layers, dilation_exponential, kernel_size)
        self.out_len = in_len - self.receptive_field + 1

        # Model blocks
        self.tcn = nn.Conv2d(
            in_channels=in_dim, 
            out_channels=h_dim, 
            kernel_size=(1, kernel_size), 
            dilation=dilation_exponential ** (n_layers - 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        x_resid = x

        # Temporal convolution layer
        h_tcn = self.tcn(x)
        h_tcn = torch.relu(h_tcn)

        _, h_dim, _, out_len = h_tcn.shape
        h_tcn = h_tcn + x_resid[:, :h_dim, :, -out_len:]

        return h_tcn