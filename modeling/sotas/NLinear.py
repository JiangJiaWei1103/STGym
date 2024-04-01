"""
Baseline method, NLinear [AAAI, 2023].
Author: ChunWei Shen

Reference:
* Paper: https://arxiv.org/abs/2205.13504
* Code: https://github.com/cure-lab/LTSF-Linear
"""
from typing import List, Any, Dict, Tuple, Optional

import torch
import torch.nn as nn
from torch import Tensor

class NLinear(nn.Module):
    """
    NLinear.

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
        super(NLinear, self).__init__()

        # Network parameters
        self.st_params = st_params
        self.out_len = out_len
        # hyperparameters of Spatial/Temporal pattern extractor
        self.t_window = self.st_params["t_window"]
        self.n_series = self.st_params["n_series"]
        self.individual = self.st_params["individual"]
        
        # Linear layer
        if self.individual:
            self.Linear = nn.ModuleList()
            for _ in range(self.n_series):
                self.Linear.append(nn.Linear(self.t_window, self.out_len))
        else:
            self.Linear = nn.Linear(self.t_window, self.out_len)

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

        x = x[..., 0]   # (B, P, N)
        seq_last = x[:,-1:,:].detach()
        x = x - seq_last

        # Linear layer
        if self.individual:
            output = torch.zeros([x.size(0), self.out_len, x.size(2)], dtype=x.dtype).to(x.device)
            for i in range(self.n_series):
                output[:, :, i] = self.Linear[i](x[:, :, i])
            x = output
        else:
            x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)

        output = x + seq_last

        return output, None, None