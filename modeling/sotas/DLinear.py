"""
Baseline method, DLinear [AAAI, 2023].
Author: ChunWei Shen

Reference:
* Paper: https://arxiv.org/abs/2205.13504
* Code: https://github.com/cure-lab/LTSF-Linear
"""
from typing import List, Any, Dict, Tuple, Optional

import torch
import torch.nn as nn
from torch import Tensor

from modeling.stgym.temporal_layers import SeriesDecompose

class DLinear(nn.Module):
    """DLinear framework.

    Parameters:
        in_len: input sequence length
        out_len: output sequence length
        n_series: number of series
        individual: whether to share linear layer for all nodes
    """
    def __init__(self, in_len: int, out_len: int, st_params: Dict[str, Any]) -> None:
        self.name = self.__class__.__name__
        super(DLinear, self).__init__()

        # Network parameters
        self.st_params = st_params
        self.out_len = out_len
        # Spatio-temporal pattern extractor
        self.n_series = self.st_params["n_series"]
        self.individual = self.st_params["individual"]
        
        # Model blocks
        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = SeriesDecompose(kernel_size)
        # Linear layer
        if self.individual:
            self.linear_seasonal = nn.ModuleList()
            self.linear_trend = nn.ModuleList()
            for _ in range(self.n_series):
                self.linear_seasonal.append(nn.Linear(in_len, self.out_len))
                self.linear_trend.append(nn.Linear(in_len, self.out_len))
        else:
            self.linear_seasonal = nn.Linear(in_len, self.out_len)
            self.linear_trend = nn.Linear(in_len, self.out_len)


    def forward(self, x: Tensor, As: Optional[List[Tensor]] = None, **kwargs: Any) -> Tuple[Tensor, None, None]:
        """Forward pass.

        Parameters:
            x: input sequence

        Shape:
            x: (B, P, N, C)
            output: (B, Q, N)
        """
        seasonal_init, trend_init = self.decompsition(x[..., 0])    # (B, P, N)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1) # (B, N, P), (B, N, P)

        # Linear layer
        if self.individual:
            seasonal_output = torch.zeros([x.size(0), x.size(2), self.out_len], dtype=x.dtype).to(x.device)
            trend_output = torch.zeros([x.size(0), x.size(2), self.out_len], dtype=x.dtype).to(x.device)
            for i in range(self.n_series):
                seasonal_output[:, i, :] = self.linear_seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.linear_trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.linear_seasonal(seasonal_init)
            trend_output = self.linear_trend(trend_init)

        x = seasonal_output + trend_output  # (B, N, out_len)
        output = x.permute(0,2,1)

        return output, None, None