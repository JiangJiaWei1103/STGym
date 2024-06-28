"""
Baseline method, LSTM [Neural Computation, 1997].
Author: ChunWei Shen

Reference:
* Paper: https://blog.xpgreat.com/file/lstm.pdf
"""
from typing import Any, Dict, List, Tuple

import torch.nn as nn
from torch import Tensor

from modeling.module.common_layers import Linear2d

class LSTM(nn.Module):
    """LSTM framework.

        Args:
            in_dim: input feature dimension
            end_dim: hidden dimension of output layer
            out_len: output sequence length
            n_layers: number of LSTM layers
            lin_h_dim: hidden dimension of input linear layer
            rnn_h_dim: hidden dimension of LSTM
            dropout: dropout ratio
    """
    
    def __init__(self, in_dim: int, end_dim: int, out_len: int, st_params: Dict[str, Any]) -> None:
        self.name = self.__class__.__name__
        super(LSTM, self).__init__()

        # Network parameters
        self.st_params = st_params
        n_layers = st_params["n_layers"]
        lin_h_dim = st_params["lin_h_dim"]
        rnn_h_dim = st_params["rnn_h_dim"]
        rnn_dropout = st_params["rnn_dropout"]
        self.out_len = out_len

        # Model blocks
        # Input linear layer
        self.in_lin = Linear2d(in_dim, lin_h_dim)
        # LSTM
        self.lstm = nn.LSTM(
            input_size=lin_h_dim, 
            hidden_size=rnn_h_dim, 
            num_layers=n_layers, 
            batch_first=True, 
            dropout=rnn_dropout
        )
        # Output layer
        self.output = nn.Sequential(nn.Linear(rnn_h_dim, end_dim), nn.ReLU(), nn.Linear(end_dim, out_len))

    def forward(self, x: Tensor, As: List[Tensor], **kwargs: Any) -> Tuple[Tensor, None, None]:
        """Forward pass.

        Args:
            x: input sequence

        Returns:
            output: prediction

        Shape:
            x: (B, P, N, C)
            output: (B, Q, N)
        """
        batch_size, t_window, n_series, n_feats = x.shape
        x = x.transpose(1, 3)   # (B, C, N, P)

        x = self.in_lin(x).permute(0, 2, 3, 1)              # (B, N, P, lin_h_dim)
        x = x.reshape(batch_size * n_series, t_window, -1)  # (B * N, P, lin_h_dim)

        h_out, _ = self.lstm(x)     # (B * N, P, rnn_h_dim)
        h = h_out[:, -1, :]         # (B * N, rnn_h_dim)

        output = self.output(h)     # (B * N, Q)
        output = output.reshape(batch_size, n_series, self.out_len).transpose(1, 2)   # (B, Q, N)

        return output, None, None