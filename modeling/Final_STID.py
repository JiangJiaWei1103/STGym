"""
STID framework.

Reference: 
https://github.com/zezhishao/STID

Author: ChunWei Shen
"""
#from typing import List, Any, Dict, Optional, Callable, Tuple
from typing import List, Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

class STID(nn.Module):
    """
    STID.

    Parameters:
        st_params: hyperparameters of Spatial/Temporal pattern extractor
        out_dim: output dimension
        priori_gs: predefined adjacency matrix
    """
    def __init__(
        self,
        st_params: Dict[str, Any],
        out_dim: int,
        priori_gs: Optional[List[Tensor]] = None,
    ):
        super(STID, self).__init__()

        # Network parameters
        self.st_params = st_params
        
        # hyperparameters of Spatial/Temporal pattern extractor
        self.num_layer = self.st_params["num_layer"]
        self.n_series = self.st_params["n_series"]
        self.t_window = self.st_params["t_window"]
        self.input_dim = self.st_params["input_dim"]
        self.node_dim = self.st_params["node_dim"]
        self.embed_dim = self.st_params["embed_dim"]   
        self.temp_dim_tid = self.st_params["temp_dim_tid"]
        self.temp_dim_diw = self.st_params["temp_dim_diw"]
        self.time_of_day_size = self.st_params["time_of_day_size"]
        self.day_of_week_size = self.st_params["day_of_week_size"]
        self.if_time_in_day = self.st_params["if_time_in_day"]
        self.if_day_in_week = self.st_params["if_day_in_week"]
        self.if_spatial = self.st_params["if_spatial"]

        self.out_dim = out_dim

        # spatial embeddings
        if self.if_spatial:
            self.node_emb = nn.Parameter(
                torch.empty(self.n_series, self.node_dim))
            nn.init.xavier_uniform_(self.node_emb)

        # temporal embeddings
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(self.time_of_day_size, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(
                torch.empty(self.day_of_week_size, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)

        # embedding layer
        self.time_series_emb_layer = nn.Conv2d(
            in_channels = self.input_dim * self.t_window, 
            out_channels = self.embed_dim,
            kernel_size = (1, 1),
            bias = True)

        # encoding
        self.hidden_dim = self.embed_dim + self.node_dim * \
            int(self.if_spatial) + self.temp_dim_tid * int(self.if_time_in_day) + \
            self.temp_dim_diw * int(self.if_day_in_week)
        self.encoder = nn.Sequential(
            *[_MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)])

        # regression
        self.regression_layer = nn.Conv2d(
            in_channels = self.hidden_dim,
            out_channels = self.out_dim,
            kernel_size = (1, 1),
            bias = True)

    def forward(
        self,
        x: Tensor,
        tid: Optional[Tensor] = None,
        diw: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[Tensor, None, None]:
        """
        Forward pass.

        Parameters:
            x: input data

        Shape:
            input: (B, T, N, C), where B is the batch_size, T is the lookback
                    time window and N is the number of time series
            tid: (B, )
            diw: (B, )
            prediction: (B, out_dim, N)

        Returns:
            prediction: prediction
        """
        # prepare data
        input_data = x[..., range(self.input_dim)]

        if self.if_time_in_day:
            t_i_d_data = x[..., 1]
            # the time_of_day feature is normalized to [0, 1]. 
            # We multiply it by 288 to get the index.
            time_in_day_emb = self.time_in_day_emb[
                (t_i_d_data[:, -1, :] * self.time_of_day_size).type(torch.LongTensor)]
        else:
            time_in_day_emb = None
        if self.if_day_in_week:
            d_i_w_data = x[..., 2]
            day_in_week_emb = self.day_in_week_emb[(
                d_i_w_data[:, -1, :]).type(torch.LongTensor)]
        else:
            day_in_week_emb = None

        # time series embedding
        batch_size, _, n_series, _ = input_data.shape
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = input_data.view(
            batch_size, n_series, -1).transpose(1, 2).unsqueeze(-1)
        time_series_emb = self.time_series_emb_layer(input_data)

        # spatial embeddings
        node_emb = []
        if self.if_spatial:
            node_emb.append(self.node_emb.unsqueeze(0).expand(
                batch_size, -1, -1).transpose(1, 2).unsqueeze(-1))  # (B, D, N, 1)
        # temporal embeddings
        tem_emb = []
        if time_in_day_emb is not None:
            tem_emb.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1))   # (B, D, N, 1)
        if day_in_week_emb is not None:
            tem_emb.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1))   # (B, D, N, 1)

        # concate all embeddings along channel
        hidden = torch.cat([time_series_emb] + node_emb + tem_emb, dim = 1)

        # encoding
        hidden = self.encoder(hidden)   # (B, D, N, 1)

        # regression
        prediction = self.regression_layer(hidden)  # (B, out_dim, N, 1)
        prediction = torch.squeeze(prediction)

        return prediction, None, None

class _MultiLayerPerceptron(nn.Module):
    """
    Multi-Layer Perceptron with residual links.

    Parameters:
        input_dim: input channel
        hidden_dim: hidden channel
    """
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int
    ):
        super(_MultiLayerPerceptron, self).__init__()

        self.fc1 = nn.Conv2d(
            in_channels = input_dim,
            out_channels = hidden_dim,
            kernel_size = (1, 1), 
            bias = True)
        self.fc2 = nn.Conv2d(
            in_channels = hidden_dim,
            out_channels = hidden_dim,
            kernel_size = (1, 1),
            bias = True)
        
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p = 0.15)

    def forward(
        self,
        input: Tensor
    ) -> Tensor:
        """
        Forward pass.

        Parameters:
            input: input data
            
        Shape:
            input: (B, C, N)
            hidden: (B, C', N)

        Returns:
            hidden: latent representation
        """

        hidden = self.fc2(self.drop(self.act(self.fc1(input))))
        hidden = hidden + input

        return hidden