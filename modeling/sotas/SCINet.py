"""
Baseline method, SCINet [NeurIPS, 2022].
Author: ChunWei Shen

Reference:
* Paper: https://arxiv.org/abs/2106.09305
* Code: https://github.com/cure-lab/SCINet
"""
from typing import List, Dict, Any, Tuple, Union, Optional

import math
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from metadata import MTSFBerks
from torch.nn.modules.loss import _Loss
from utils.scaler import MaxScaler, StandardScaler

from modeling.module.layers import SCINetTree

class SCINet(nn.Module):
    """SCINet framework.

        Args:
            in_len: input sequence length
            out_len: output sequence length
            n_series: number of series
            n_stacks: number of SCINet block
            n_levels: number of SCINet block levels
            n_decoder_layer: number of decoder layers
            h_ratio: in_dim * h_ratio = hidden dimension
            kernel_size: kernel size
            groups: groups
            dropout: dropout ratio
            INN: if True, apply interactive learning
            positionalEcoding: if True, apply positional ecoding
            dataset_name: dataset name
            lastWeight: loss weight for final step
            criterion: criterion
    """
    
    def __init__(
        self,
        in_len: int,
        out_len: int,
        n_series: int,
        st_params: Dict[str, Any],
        loss_params: Dict[str, Any]
    ) -> None:
        self.name = self.__class__.__name__
        super(SCINet, self).__init__()

        # Network parameters
        self.st_params = st_params
        self.loss_params = loss_params
        # Spatio-temporal pattern extractor
        self.n_stacks  = st_params["n_stacks"]
        self.n_levels = st_params["n_levels"]
        self.n_decoder_layer = st_params["n_decoder_layer"]
        h_ratio = st_params["h_ratio"]
        kernel_size = st_params["kernel_size"]
        groups = st_params["groups"]
        dropout = st_params["dropout"]
        INN = st_params["INN"]
        self.pe = st_params["positional_ecoding"]
        self.dataset_name = st_params["dataset_name"]
        # hyperparameters of loss function
        self.single_step = loss_params["single_step"]
        self.lastWeight = loss_params["lastWeight"]
        criterion = loss_params["criterion"]
        self.concat_len = in_len - out_len
        self.in_len = in_len
        self.n_series = n_series
        self.criterion = self.smooth_l1_loss if criterion == "smooth_l1_loss" else None

        # Model blocks
        # Encoder Tree
        self.blocks1 = Encoder(
            in_dim=n_series,
            h_ratio=h_ratio,
            kernel_size=kernel_size,
            groups=groups,
            dropout=dropout,
            INN=INN,
            n_levels=self.n_levels
        )
        if self.n_stacks == 2:
            self.blocks2 = Encoder(
                in_dim=n_series,
                h_ratio=h_ratio,
                kernel_size=kernel_size,
                groups=groups,
                dropout=dropout,
                INN=INN,
                n_levels=self.n_levels
            )

        # Output layers
        self.div_proj = nn.ModuleList()
        self.overlap_len = in_len // 4
        self.div_len = in_len // 6
        if self.n_decoder_layer > 1:
            self.output1 = nn.Linear(in_features=in_len, out_features=out_len)
            for _ in range(self.n_decoder_layer - 1):
                div_proj = nn.ModuleList()
                for i in range(6):
                    lens = min(i * self.div_len + self.overlap_len, in_len) - i * self.div_len
                    div_proj.append(nn.Linear(in_features=lens, out_features=self.div_len))
                self.div_proj.append(div_proj)
        else:
            self.output1 = nn.Conv1d(in_channels=in_len, out_channels=out_len, kernel_size=1, stride=1, bias=False)

        if self.n_stacks == 2:
            if self.concat_len:
                self.output2 = nn.Conv1d(
                    in_channels=self.concat_len + out_len,
                    out_channels=out_len,
                    kernel_size=1,
                    bias=False
                )
            else:
                self.output2 = nn.Conv1d(
                    in_channels=in_len + out_len,
                    out_channels=out_len,
                    kernel_size=1,
                    bias=False
                )

    def forward(
        self, x: Tensor, As: Optional[List[Tensor]] = None, **kwargs: Any
    ) -> Tuple[Tensor, Union[Tensor, None], None]:
        """Forward pass.

        Args:
            x: input sequence

        Shape:
            x: (B, P, N, C)
        """
        x = x[..., 0]
        # evenly divided the input length into two parts. (e.g., 32 -> 16 -> 8 -> 4 for 3 levels)
        assert self.in_len % (np.power(2, self.n_levels)) == 0

        # Positional encoding
        if self.pe:
            pe = self.get_position_encoding(x)
            if pe.shape[2] > x.shape[2]:
                x += pe[:, :, :-1]
            else:
                x += self.get_position_encoding(x)

        # SCI-Block
        resid = x
        x = self.blocks1(x)
        x += resid
        # Decoder layers
        if self.n_decoder_layer == 1:
            x = self.output1(x)
        else:
            x = x.permute(0,2,1)
            for div_proj in self.div_proj:
                output = torch.zeros(x.shape, dtype = x.dtype).to(x.device)
                for i, div_layer in enumerate(div_proj):
                    div_x = x[:, :, i * self.div_len : min(i * self.div_len + self.overlap_len, self.in_len)]
                    output[:, :, i * self.div_len : (i + 1) * self.div_len] = div_layer(div_x)
                x = output
            x = self.output1(x)
            x = x.permute(0,2,1)

        if self.n_stacks == 1:
            return x, None, None
        elif self.n_stacks == 2:
            midoutput = x
            if self.concat_len:
                x = torch.cat((resid[:, -self.concat_len:,:], x), dim=1)
            else:
                x = torch.cat((resid, x), dim=1)
            resid = x
            x = self.blocks2(x)
            x += resid
            x = self.output2(x)

            return x, midoutput, None
        
    def get_position_encoding(self, x: Tensor) -> Tensor:
        """Positional encoding.

        Args:
            x: input sequence

        Shape:
            x: (B, L, N)
            encoding: (1, L, pe_h_dim)
        """
        self.pe_h_dim = self.n_series
        if self.pe_h_dim % 2 == 1:
            self.pe_h_dim += 1
    
        num_timescales = self.pe_h_dim // 2
        max_timescale = 10000.0
        min_timescale = 1.0

        log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / max(num_timescales - 1, 1))
        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales, dtype=torch.float32) * -log_timescale_increment).to(x.device)
        
        max_length = x.size()[1]
        position = torch.arange(max_length, dtype=torch.float32, device=x.device)
        scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)                # (max_length, num_timescales)
        encoding = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim = 1)  # (T, N) or (T, N + 1)
        encoding = F.pad(encoding, (0, 0, 0, self.pe_h_dim % 2))
        encoding = encoding.view(1, max_length, self.pe_h_dim)
    
        return encoding
    
    def smooth_l1_loss(self, y_pred: Tensor, y_true: Tensor, beta: float = 1./9, size_average: bool = True) -> float:
        """Smooth L1 loss.

        Very similar to the smooth_l1_loss from pytorch,
        but with the extra beta parameter.
        """
        n = torch.abs(y_pred - y_true)
        cond = n < beta
        loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)

        if size_average:
            return loss.mean()
        
        return loss.sum()
        
    def train_loss(
        self,
        y_pred: Tensor,
        y_true: Tensor,
        aux_output: List[Tensor],
        scaler: Union[MaxScaler, StandardScaler],
        criterion: _Loss,
    ) -> float:
        """Custom loss function.

        Args:
            y_pred: model prediction
            y_true: ground truth
            aux_output: auxiliary output
            scaler: scaler
        """
        if self.criterion is None:
            self.criterion = criterion
            
        weight = torch.tensor(self.lastWeight).to(y_pred.device)

        midoutput = aux_output[0]
        if midoutput is not None:
            midoutput = scaler.inverse_transform(midoutput)

        if self.single_step:
            y_last = y_true[:, -1, :]
            loss_f = self.criterion(y_pred[:, -1, :], y_last)
            if self.n_stacks == 2:
                loss_m = self.criterion(midoutput, y_true) / midoutput.shape[1] # average results
        else:
            if self.lastWeight == 1.0:
                loss_f = self.criterion(y_pred, y_true)
                if self.n_stacks == 2:
                    loss_m = self.criterion(midoutput, y_true)
            else:
                loss_f = self.criterion(y_pred[:, :-1, :], y_true[:, :-1, :] ) \
                        + weight * self.criterion(y_pred[:, -1:, :], y_true[:, -1:, :] )
                if self.n_stacks == 2:
                    loss_m = self.criterion(midoutput[:, :-1, :] , y_true[:, :-1, :] ) \
                            + weight * self.criterion(midoutput[:, -1:, :], y_true[:, -1:, :])
                    
        loss = loss_f
        if self.n_stacks == 2:
            loss += loss_m
        
        return loss
    
    def eval_loss(
        self,
        y_pred: Tensor,
        y_true: Tensor,
        aux_output: List[Tensor],
        scaler: Union[MaxScaler, StandardScaler],
        criterion: _Loss,
    ) -> Tuple[float, Tensor, Tensor]:
        """Custom loss function.

        Args:
            y_pred: model prediction
            y_true: ground truth
            aux_output: auxiliary output
            scaler: scaler
            criterion: criterion
        """
        y_pred_inv = scaler.inverse_transform(y_pred)
        y_true_inv = scaler.inverse_transform(y_true)
        midoutput = aux_output[0]
        if midoutput is not None:
            midoutput = scaler.inverse_transform(midoutput)

        if self.dataset_name in MTSFBerks:
            y_pred = y_pred[:, -1, :]
            y_true = y_true[:, -1, :]
            y_pred_inv = y_pred_inv[:, -1, :]
            y_true_inv = y_true_inv[:, -1, :]
            loss = criterion(y_pred_inv, y_true_inv)
        else:
            loss = criterion(y_pred_inv, y_true_inv)

        return loss, y_pred, y_true
    

class Encoder(nn.Module):
    """Encoder Tree."""

    def __init__(
        self, in_dim: int, h_ratio: int, kernel_size: int, groups: int, dropout: float, INN: bool, n_levels: int
    ) -> None:
        super(Encoder, self).__init__()

        self.SCINet_Tree = SCINetTree(
            in_dim=in_dim,
            h_ratio=h_ratio,
            kernel_size=kernel_size,
            groups=groups,
            dropout=dropout,
            INN=INN,
            current_level=n_levels-1
        )
        
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: input sequence

        Shape:
            x: (B, L, N)
            output: (B, L, N)
        """
        output = self.SCINet_Tree(x)

        return output