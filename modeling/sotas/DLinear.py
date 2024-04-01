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

class DLinear(nn.Module):
    """
    DLinear.

    Parameters:
        t_window: lookback time window
        n_series: number of nodes
        individual: whether to share linear layer for all nodes
        out_len: output sequence length
    """
    def __init__(
        self,
        st_params: Dict[str, Any],
        out_len: int
    ):
        super(DLinear, self).__init__()

        # Network parameters
        self.st_params = st_params
        self.out_len = out_len
        # hyperparameters of Spatial/Temporal pattern extractor
        self.t_window = self.st_params["t_window"]
        self.n_series = self.st_params["n_series"]
        self.individual = self.st_params["individual"]
        
        
        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = _Series_Decomp(kernel_size)

        # Linear layer
        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for _ in range(self.n_series):
                self.Linear_Seasonal.append(nn.Linear(self.t_window, self.out_len))
                self.Linear_Trend.append(nn.Linear(self.t_window, self.out_len))
        else:
            self.Linear_Seasonal = nn.Linear(self.t_window, self.out_len)
            self.Linear_Trend = nn.Linear(self.t_window, self.out_len)


    def forward(
        self,
        x: Tensor,
        As: Optional[List[Tensor]] = None,
        **kwargs: Any,
    ) -> Tuple[Tensor, None, None]:
        """
        Forward pass.

        Parameters:
            x: input features

        Shape:
            x: (B, P, N, C)
            output: (B, Q, N)
        """
        seasonal_init, trend_init = self.decompsition(x[..., 0])    # (B, P, N)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1) # (B, N, P), (B, N, P)

        # Linear layer
        if self.individual:
            seasonal_output = torch.zeros(
                [seasonal_init.size(0), seasonal_init.size(1), self.out_len],
                dtype=seasonal_init.dtype
            ).to(seasonal_init.device)

            trend_output = torch.zeros(
                [trend_init.size(0), trend_init.size(1), self.out_len],
                dtype=trend_init.dtype
            ).to(trend_init.device)

            for i in range(self.n_series):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output  # (B, N, out_len)

        return x.permute(0,2,1), None, None
    
class _Moving_Avg(nn.Module):
    """
    Moving average block to highlight the trend of time series.

    Parameters:
        kernel_size: window size of moving average
        stride: stride of moving average
    """
    def __init__(
        self,
        kernel_size: int,
        stride: int
    ):
        super(_Moving_Avg, self).__init__()

        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(
        self,
        x: Tensor
    ) -> Tensor:
        
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)

        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)

        return x
    
class _Series_Decomp(nn.Module):
    """
    Series decomposition block.

    Parameters:
        kernel_size: window size of moving average
    """
    def __init__(
        self,
        kernel_size: int
    ):
        super(_Series_Decomp, self).__init__()
        self.moving_avg = _Moving_Avg(kernel_size, stride=1)

    def forward(
        self,
        x: int
    ):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean

        return res, moving_mean