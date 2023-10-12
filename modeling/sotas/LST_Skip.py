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
    """
    LST-Skip.

    Parameters:
        hidRNN: hidden dimension of RNN
        hidCNN: hidden dimension of CNN
        hidSkip: hidden dimension of Skip-RNN
        kernel_size: kernel size
        skip: number of hidden cells skipped through
        highway_window: time window for Autoregressive component
        t_window: lookback time window
        n_series: number of nodes
        dropout: dropout ratio
        output_act: activation function of the final output
        out_len: output sequence length
    """
    def __init__(
        self,
        st_params: Dict[str, Any],
        out_len: int
    ):
        super(LST_Skip, self).__init__()

        # Network parameters
        self.st_params = st_params
        self.out_len = out_len
        # hyperparameters of Spatial/Temporal pattern extractor
        self.hidR = self.st_params['hidRNN']
        self.hidC = self.st_params['hidCNN']
        self.hidS = self.st_params['hidSkip']
        self.k = self.st_params['kernel_size']
        self.skip = self.st_params['skip']
        self.hw = self.st_params['highway_window']
        self.t_window = self.st_params['t_window']
        self.n_series = self.st_params['n_series']
        dropout_ratio = self.st_params['dropout']
        output_act = self.st_params['output_act']

        self.pt = math.floor((self.t_window - self.k) / self.skip)

        # CNN, Convolutional Component
        self.conv = nn.Conv2d(
            1,
            self.hidC,
            kernel_size = (self.k, self.n_series))
        
        # RNN, Recurrent Component
        self.GRU = nn.GRU(self.hidC, self.hidR)

        self.dropout = nn.Dropout(p = dropout_ratio)

        # Skip-RNN, Recurrent-skip Component
        if (self.skip > 0):
            self.GRUskip = nn.GRU(self.hidC, self.hidS)
            self.linear = nn.Linear(self.hidR + self.skip * self.hidS, self.n_series)
        else:
            self.linear = nn.Linear(self.hidR, self.n_series)

        # Autoregressive Component
        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, out_len)
            
        self.output = None
        if (output_act == 'sigmoid'):
            self.output = F.sigmoid
        if (output_act == 'tanh'):
            self.output = F.tanh
 
    def forward(
        self,
        x: Tensor,
        As: Optional[List[Tensor]] = None,
        **kwargs: Any,
    ) -> Tuple[Tensor, None, None]:
        """
        Forward pass.

        Parameters:
            x: input hour features

        Shape:
            x: (B, T, N, C), where B is the batch_size, T is the lookback
                time window and N is the number of time series
            output: (B, N)
        """
        batch_size = x.size(0)
        
        # CNN, Convolutional Component
        c = x.permute(0, 3, 1, 2)   # (B, C, T, N)
        c = F.relu(self.conv(c))    # (B, hidC, T', 1)
        c = self.dropout(c)
        c = torch.squeeze(c, 3)     # (B, hidC, T')
        
        # RNN, Recurrent Component
        r = c.permute(2, 0, 1).contiguous()     # (T', B, hidC)
        _, r = self.GRU(r)
        r = self.dropout(torch.squeeze(r, 0))   # (B, hidR)

        
        # Skip-RNN, Recurrent-skip Component
        if (self.skip > 0):
            s = c[:,:, int(-self.pt * self.skip):].contiguous()
            s = s.view(batch_size, self.hidC, self.pt, self.skip)
            s = s.permute(2,0,3,1).contiguous()
            s = s.view(self.pt, batch_size * self.skip, self.hidC)
            _, s = self.GRUskip(s)
            s = s.view(batch_size, self.skip * self.hidS)
            s = self.dropout(s)
            r = torch.cat((r, s), 1)    # (B, hidR+skip*hidS)
        
        res = self.linear(r)    # (B, N)
        
        # Autoregressive Component
        if (self.hw > 0):
            z = x.squeeze(-1)[:, -self.hw:, :]                      # (B, hw, N)
            z = z.permute(0,2,1).contiguous().view(-1, self.hw)     # (B*N, hw)
            z = self.highway(z)                                     # (B*N, 1)
            z = z.view(-1,self.n_series)                            # (B, N)
            res = res + z                                           # (B, N)
            
        if (self.output):
            res = self.output(res)

        return res, None, None