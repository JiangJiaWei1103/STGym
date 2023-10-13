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


class TPA_LSTM(nn.Module):
    """
    TPA-LSTM.

    Parameters:
        n_series: number of nodes
        t_window: lookback time window
        linear_h_dim: hidden dimension of input linear projection
        rnn_h_dim: hidden dimension of rnn
        rnn_n_layers: number of rnn layers
        rnn_dropout: dropout ratio of rnn
        conv_out_ch: convolution output channels
        ar_window: autoregressive lookback time window
        out_len: output dimension
    """

    def __init__(
        self,
        net_params: Dict[str, Any],
        rnn_params: Dict[str, Any],
        cnn_params: Dict[str, Any],
        ar_params: Dict[str, Any],
        out_len: int
    ):
        self.name = self.__class__.__name__
        super(TPA_LSTM, self).__init__()

        # Network parameters
        self.net_params = net_params
        self.rnn_params = rnn_params
        self.cnn_params = cnn_params
        self.ar_params = ar_params
        # hyperparameters of Network
        n_series = self.net_params['n_series']
        t_window = self.net_params['t_window']
        # hyperparameters of RNN
        linear_h_dim = self.rnn_params['linear_h_dim']
        rnn_h_dim = self.rnn_params['rnn_h_dim']
        self.rnn_n_layers = self.rnn_params['rnn_n_layers']
        rnn_dropout = self.rnn_params['rnn_dropout']
        # hyperparameters of CNN
        conv_out_ch = self.cnn_params['conv_out_ch']
        # hyperparameters of Autoregressive
        self.ar_window = self.ar_params['ar_window']
        # Common
        self.rnn_tpa_cat_size = rnn_h_dim * self.rnn_n_layers + rnn_h_dim

        # Model blocks
        # RNN
        self.l_in = nn.Linear(n_series, linear_h_dim)
        self.rnn = nn.LSTM(
            linear_h_dim,
            rnn_h_dim,
            num_layers=self.rnn_n_layers,
            batch_first=True,
            dropout=rnn_dropout,
        )

        # CNN, temporal pattern extractor
        self.t_extr = nn.Conv1d(t_window - 1, conv_out_ch, kernel_size=1)

        # TPA, temporal pattern attention mechanism
        self.tpa_linear = nn.Linear(rnn_h_dim * self.rnn_n_layers, conv_out_ch)
        self.tpa = _TemporalPatternAttention(
            embed_dim=rnn_h_dim, conv_out_ch=conv_out_ch
        )

        # Autoregressive linear modeling
        self.ar_linear = nn.Linear(self.ar_window, out_len)
        
        # Common
        self.output = nn.Linear(self.rnn_tpa_cat_size, n_series)

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
        
        Return:
            output: prediction

        Shape:
            x: (B, P, N, C)
            output: (B, Q)
        """
        x = x[..., 0]
        batch_size = x.size(0)  # B
        x_l = x[:, -self.ar_window :, :].transpose(1, 2)  # (B, n_series, ar_window)

        # RNN
        x_r = F.relu(self.l_in(x))
        h_out, (h_t, c_t) = self.rnn(x_r)           # (B, P, rnn_h_dim)
        if self.rnn_n_layers == 1:
            h_t = torch.squeeze(h_t, dim=0)         # (B, rnn_h_dim)
        else:
            h_t = h_t.transpose(0, 1).contiguous()
            h_t = h_t.view(batch_size, -1)          # (B, rnn_h_dim * rnn_n_layers)

        # CNN, temporal pattern extractor
        h_out = h_out[:, :-1, :]
        h_c = self.t_extr(h_out)                    # (B, conv_out_ch, rnn_h_dim)

        # TPA, temporal pattern attention mechanism
        h_t_tpa = self.tpa_linear(h_t)              # (B, conv_out_ch)
        query = torch.unsqueeze(h_t_tpa, dim=1)     # (B, 1, conv_out_ch)
        h_c = h_c.permute(0, 2, 1)                  # (B, rnn_h_dim, conv_out_ch)
        tpa_out = self.tpa(
            query=query, key=h_c, value=h_c
        )                                           # (B, rnn_h_dim)

        # Autoregressive linear modeling
        x_l = self.ar_linear(x_l)                   # (B, n_series, 1)
        x_l = torch.squeeze(x_l)                    # (B, n_series)

        # Output
        x_nl = torch.cat((h_t, tpa_out), dim=1)     # (B, rnn_tpa_cat_size)
        x_nl = self.output(x_nl)                    # (B, n_series)
        output = x_l + x_nl

        return output, None, None

class _TemporalPatternAttention(nn.Module):
    def __init__(
        self, 
        embed_dim: int,
        conv_out_ch: int,
    ):
        """
        Temporal Pattern Attention.

        Parameters:
            embed_dim: number of output dimension of tpa
            conv_out_ch: number of output channels of temporal convolution
        """
        super(_TemporalPatternAttention, self).__init__()

        self.fc = nn.Linear(conv_out_ch * 2, embed_dim)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor
    ) -> Tensor:
        """
        Forward pass.

        Parameters:
            query: query for temporal pattern attention
            key: key for temporal pattern attention
            value: value for temporal pattern attention

        Shape:
            query: (B, 1, conv_out_ch)
            key: (B, conv_out_ch, rnn_h_dim)
            value: (B, conv_out_ch, rnn_h_dim)
            output: (B, embed_dim)
        """

        # attention weight
        a = torch.sigmoid(torch.sum(key * query, dim=2))    # (B, rnn_h_dim)

        # context vector
        v = torch.sum(a.unsqueeze(-1) * value, dim=1)       # (B, conv_out_ch)

        output = self.fc(torch.cat([query.squeeze(1), v], dim=1))

        return output