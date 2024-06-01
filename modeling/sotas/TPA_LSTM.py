"""
Baseline method, TPA-LSTM [ECML/PKDD, 2019].
Author: ChunWei Shen

Reference:
* Paper: https://arxiv.org/abs/1809.04206
* Code: https://github.com/shunyaoshih/TPA-LSTM
"""
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from modeling.stgym.temporal_layers import TemporalPatternAttention


class TPA_LSTM(nn.Module):
    """TPA-LSTM framework.

    Parameters:
        in_len: input sequence length
        out_len: output sequence length
        n_series: number of series
        lin_h_dim: hidden dimension of input linear layer
        rnn_h_dim: hidden dimension of RNN
        rnn_n_layers: number of RNN layers
        rnn_dropout: dropout ratio of RNN
        cnn_h_dim: hidden dimension of CNN
        ar_window: autoregressive lookback time window
    """
    def __init__(self, in_len: int, out_len: int, n_series: int, st_params: Dict[str, Any]) -> None:
        self.name = self.__class__.__name__
        super(TPA_LSTM, self).__init__()

        # Network parameters
        self.st_params = st_params
        # Spatio-temporal pattern extractor
        lin_h_dim = st_params["lin_h_dim"]
        rnn_h_dim = st_params["rnn_h_dim"]
        self.rnn_n_layers = st_params["rnn_n_layers"]
        rnn_dropout = st_params["rnn_dropout"]
        cnn_h_dim = st_params["cnn_h_dim"]
        self.ar_window = st_params['ar_window']
        self.cat_dim = rnn_h_dim * self.rnn_n_layers + rnn_h_dim

        # Model blocks
        # Input linear layer
        self.in_lin = nn.Linear(n_series, lin_h_dim)
        # RNN
        self.rnn = nn.LSTM(
            lin_h_dim,
            rnn_h_dim,
            num_layers=self.rnn_n_layers,
            batch_first=True,
            dropout=rnn_dropout,
        )
        # CNN
        self.cnn = nn.Conv1d(in_len - 1, cnn_h_dim, kernel_size=1)
        # TPA, temporal pattern attention mechanism
        self.tpa_linear = nn.Linear(rnn_h_dim * self.rnn_n_layers, cnn_h_dim)
        self.tpa = TemporalPatternAttention(h_dim=cnn_h_dim, out_dim=rnn_h_dim)
        # Autoregressive linear modeling
        self.ar_linear = nn.Linear(self.ar_window, out_len)
        # Output layer
        self.output = nn.Linear(self.cat_dim, n_series)

    def forward(self, x: Tensor, As: Optional[List[Tensor]] = None, **kwargs: Any) -> Tuple[Tensor, None, None]:
        """Forward pass.

        Parameters:
            x: input sequence

        Shape:
            x: (B, P, N, C)
            output: (B, Q)
        """
        batch_size = x.size(0)

        x = x[..., 0]
        x_ar = x[:, -self.ar_window:, :].transpose(1, 2)  # (B, n_series, ar_window)

        # RNN
        h_r = F.relu(self.in_lin(x))
        h_out, (h_t, c_t) = self.rnn(h_r)           # (B, P, rnn_h_dim)
        if self.rnn_n_layers == 1:
            h_t = torch.squeeze(h_t, dim=0)         # (B, rnn_h_dim)
        else:
            h_t = h_t.transpose(0, 1).contiguous()
            h_t = h_t.view(batch_size, -1)          # (B, rnn_h_dim * rnn_n_layers)

        # CNN
        h_out = h_out[:, :-1, :]
        h_c = self.cnn(h_out)                       # (B, cnn_h_dim, rnn_h_dim)

        # TPA, temporal pattern attention mechanism
        h_tpa = self.tpa_linear(h_t)                # (B, cnn_h_dim)
        query = torch.unsqueeze(h_tpa, dim=1)       # (B, 1, cnn_h_dim)
        h_c = h_c.permute(0, 2, 1)                  # (B, rnn_h_dim, cnn_h_dim)
        tpa_out = self.tpa(                         # (B, rnn_h_dim)
            query=query, key=h_c, value=h_c
        ).squeeze(dim=1)

        # Autoregressive linear modeling
        h_a = self.ar_linear(x_ar).squeeze(dim=-1)  # (B, n_series)

        # Output layer∂
        h = torch.cat((h_t, tpa_out), dim=1)     # (B, rnn_tpa_cat_size)
        h = self.output(h)                       # (B, n_series)
        output = h_a + h

        return output, None, None