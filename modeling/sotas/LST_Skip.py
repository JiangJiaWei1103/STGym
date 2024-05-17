"""
Baseline method, LST-Skip [SIGIR, 2018].
Author: ChunWei Shen

Reference:
* Paper: https://arxiv.org/abs/1703.07015
* Code: https://github.com/laiguokun/LSTNet
"""
from typing import List, Dict, Any, Optional, Tuple

import math
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

class LST_Skip(nn.Module):
    """LST-Skip framework.

    Parameters:
        in_dim: input dimension
        in_len: input sequence length
        out_len: output sequence length
        n_series: number of series
        rnn_h_dim: hidden dimension of RNN
        cnn_h_dim: hidden dimension of CNN
        skip_h_dim: hidden dimension of Skip-RNN
        kernel_size: kernel size
        n_skip: number of hidden cells skipped through
        ar_window: autoregressive lookback time window
        dropout: dropout ratio
        act: activation function of the output layer
    """
    def __init__(self, in_dim: int, in_len: int, out_len: int, n_series: int, st_params: Dict[str, Any]) -> None:
        self.name = self.__class__.__name__
        super(LST_Skip, self).__init__()

        # Network parameters
        self.st_params = st_params
        # Spatio-temporal pattern extractor
        rnn_h_dim = st_params["rnn_h_dim"]
        self.cnn_h_dim = st_params["cnn_h_dim"]
        self.skip_h_dim = st_params["skip_h_dim"]
        kernel_size = st_params["kernel_size"]
        self.n_skip = st_params["n_skip"]
        self.ar_window = st_params["ar_window"]
        dropout = st_params["dropout"]
        act = st_params["act"]
        self.n_series = n_series
        self.pt = math.floor((in_len - kernel_size) / self.n_skip)

        # Model blocks
        # CNN
        self.conv = nn.Conv2d(in_channels=in_dim, out_channels=self.cnn_h_dim, kernel_size=(kernel_size, n_series))
        # RNN
        self.GRU = nn.GRU(self.cnn_h_dim, rnn_h_dim)
        self.dropout = nn.Dropout(p=dropout)
        # Recurrent-skip Component
        if (self.n_skip > 0):
            self.GRUskip = nn.GRU(self.cnn_h_dim, self.skip_h_dim)
            self.linear = nn.Linear(rnn_h_dim + self.n_skip * self.skip_h_dim, n_series)
        else:
            self.linear = nn.Linear(rnn_h_dim, n_series)
        # Autoregressive Component
        if (self.ar_window > 0):
            self.ar_linear = nn.Linear(self.ar_window, out_len)
            
        if act is not None:
            if act == "sigmoid":
                self.act = F.sigmoid
            if act == "tanh":
                self.act = F.tanh
        else:
            self.act = None
 
    def forward(self, x: Tensor, As: Optional[List[Tensor]] = None, **kwargs: Any) -> Tuple[Tensor, None, None]:
        """Forward pass.

        Parameters:
            x: input sequence

        Shape:
            x: (B, P, N, C)
            output: (B, Q)
        """
        batch_size = x.size(0)
        
        # CNN
        h_c = x.permute(0, 3, 1, 2)   # (B, C, P, N)
        h_c = self.dropout(F.relu(self.conv(h_c))).squeeze(dim=-1)    # (B, cnn_h_dim, L)
        
        # RNN
        h_r = h_c.permute(2, 0, 1).contiguous()     # (L, B, cnn_h_dim)
        _, h_r = self.GRU(h_r)
        h_r = self.dropout(h_r.squeeze(dim=0))      # (B, rnn_h_dim)
        
        # Recurrent-skip Component
        if (self.n_skip > 0):
            h_s = h_c[:,:, int(-self.pt * self.n_skip):].contiguous()
            h_s = h_s.view(batch_size, self.cnn_h_dim, self.pt, self.n_skip)
            h_s = h_s.permute(2,0,3,1).contiguous()
            h_s = h_s.view(self.pt, batch_size * self.n_skip, self.cnn_h_dim)
            _, h_s = self.GRUskip(h_s)
            h_s = h_s.view(batch_size, self.n_skip * self.skip_h_dim)
            h_s = self.dropout(h_s)
            h = torch.cat((h_r, h_s), 1)    # (B, rnn_h_dim + n_skip * skip_h_dim)
        output = self.linear(h)    # (B, N)
        
        # Autoregressive Component
        if (self.ar_window > 0):
            h_a = x.squeeze(-1)[:, -self.ar_window:, :]                      # (B, hw, N)
            h_a = h_a.permute(0,2,1).contiguous().view(-1, self.ar_window)   # (B * N, hw)
            h_a = self.ar_linear(h_a)                                        # (B * N, 1)
            h_a = h_a.view(-1, self.n_series)                                # (B, N)
            output = output + h_a                                            # (B, N)
            
        if self.act is not None:
            output = self.act(h)

        return output, None, None