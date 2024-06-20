"""
Baseline method, STGCN [IJCAI, 2018].
Author: ChunWei Shen

Reference:
* Paper: https://arxiv.org/abs/1709.04875
* Code: https://github.com/hazdzz/STGCN
"""
from typing import List, Any, Dict, Tuple

import torch.nn as nn
from torch import Tensor

from modeling.module.layers import STConvBlock
from modeling.module.temporal_layers import TemporalConvLayer

class STGCN(nn.Module):
    """STGCN framework.

    Parameters:
        in_dim: input feature dimension
        in_len: input sequence length
        end_dim: hidden dimension of output layer
        out_len: output sequence length
        st_h_dims: hidden dimension of STBlock
        kernel_size: kernel size
        n_series: number of nodes
        cheb_k: order of chebyshev polynomial expansion
        act: activation function
        dropout: dropout ratio
    """
    
    def __init__(self, in_dim: int, in_len: int, end_dim: int, out_len: int, st_params: Dict[str, Any]) -> None:
        self.name = self.__class__.__name__
        super(STGCN, self).__init__()

        # Network parameters
        self.st_params = st_params
        # Spatio-temporal pattern extractor
        self.st_h_dims = st_params["st_h_dims"]
        kernel_size = st_params["kernel_size"]
        n_series = st_params["n_series"]
        cheb_k = st_params["cheb_k"]
        act = st_params["act"]
        dropout = st_params["dropout"]
        self.in_len = in_len

        # Stacked spatio-temporal blocks
        self.st_blocks = nn.ModuleList()
        for l in range(len(self.st_h_dims)):
            in_dim = in_dim if l == 0 else self.st_h_dims[l - 1][-1]
            self.st_blocks.append(
                STConvBlock(
                    in_dim=in_dim,
                    h_dims=self.st_h_dims[l],
                    n_series=n_series,
                    kernel_size=kernel_size,
                    cheb_k=cheb_k,
                    act=act,
                    dropout=dropout
                )
            )

        # output layer
        self.receptive_field = len(self.st_h_dims) * 2 * (kernel_size - 1)
        if in_len > self.receptive_field:
            self.tcn = TemporalConvLayer(
                in_channels=self.st_h_dims[-1][-1], 
                out_channels=end_dim, 
                kernel_size=in_len - self.receptive_field,
                act=act
            )
            self.ln = nn.LayerNorm([n_series, end_dim])
            self.output = nn.Sequential(
                nn.Linear(in_features=end_dim, out_features=end_dim),
                nn.ReLU(),
                nn.Linear(in_features=end_dim, out_features=out_len)
            )
        elif in_len == self.receptive_field:
            self.output = nn.Sequential(
                nn.Linear(in_features=self.st_h_dims[-1][-1], out_features=end_dim),
                nn.ReLU(),
                nn.Linear(in_features=end_dim, out_features=out_len)
            )

    def forward(self, x: Tensor, As: List[Tensor], **kwargs: Any) -> Tuple[Tensor, None, None]:
        """
        Forward pass.

        Parameters:
            input: input features
            As: list of adjacency matrices

        Shape:
            input: (B, P, N, C)
            output: (B, Q, N)
        """
        x = x.permute(0, 3, 2, 1)               # (B, C, N, P)

        # Stacked spatio-temporal blocks
        for i in range(len(self.st_h_dims)):
            x = self.st_blocks[i](x, As)        # (B, C, N, L)

        # output layer
        if self.in_len > self.receptive_field:
            output = self.tcn(x).permute(0, 3, 2, 1)    # (B, 1, N, C)
            output = self.ln(output)
            output = self.output(output)                # (B, 1, N, Q)
        elif self.in_len == self.receptive_field:
            x = x.permute(0, 3, 2, 1)                   # (B, 1, N, Q)
            output = self.output(x)

        output = output.permute(0, 3, 2, 1).squeeze(dim=-1) # (B, Q, N)
        
        return output, None, None