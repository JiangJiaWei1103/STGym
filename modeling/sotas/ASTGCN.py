"""
Baseline method, ASTGCN [AAAI, 2019].
Author: ChunWei Shen

Reference:
* Paper: https://ojs.aaai.org/index.php/AAAI/article/view/3881
* Code: https://github.com/guoshnBJTU/ASTGCN-r-pytorch
"""
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

import numpy as np

class ASTGCN(nn.Module):
    """
    ASTGCN.

    Parameters:
        num_block: number of ASTGCN block
        in_channels: dimension of input channel
        K: oreder of chebyshev graph convolution
        num_chev_filter: output dimension of Chebyshev convolution
        num_time_filter: output dimension of ASTGCN block
        t_window: lookback time window (hour)
        day_window: lookback time window (day)
        week_window: lookback time window (week)
        num_of_hour: number of hours for lookback window
        num_of_day: number of days for lookback window
        num_of_week: number of week for lookback window
        n_series: number of nodes
        device: device
        out_len: output sequence length
    """
    def __init__(
        self,
        st_params: Dict[str, Any],
        device: str,
        out_len: int,
    ) -> None:
        super(ASTGCN, self).__init__()

        # Network parameters
        self.st_params = st_params
        self.out_len = out_len
        # hyperparameters of Spatial/Temporal Convolution Module
        num_block = self.st_params['num_block']
        in_channels = self.st_params['in_channels']
        self.K = self.st_params['K']
        num_chev_filter = self.st_params['num_chev_filter']
        num_time_filter = self.st_params['num_time_filter']
        t_window = self.st_params['t_window']
        day_window = self.st_params['day_window']
        week_window = self.st_params['week_window']
        num_of_hour = self.st_params['num_of_hour']
        num_of_day = self.st_params['num_of_day']
        num_of_week = self.st_params['num_of_week']
        n_series = self.st_params['n_series']
        
        self.astgcn = nn.ModuleList()
        # ASTGCN submodule for recent data
        self.astgcn.append(
            _ASTGCN_submodule(
                num_block,
                in_channels,
                self.K,
                num_chev_filter,
                num_time_filter,
                num_of_hour,
                self.out_len,
                t_window,
                n_series,
                device))
        
        # ASTGCN submodule for daily-periodic data
        self.astgcn.append(
            _ASTGCN_submodule(
                num_block,
                in_channels,
                self.K,
                num_chev_filter,
                num_time_filter,
                num_of_day,
                self.out_len,
                day_window,
                n_series,
                device))
        
        # ASTGCN submodule for weekly-periodic data
        self.astgcn.append(
            _ASTGCN_submodule(
                num_block,
                in_channels,
                self.K,
                num_chev_filter,
                num_time_filter,
                num_of_week,
                self.out_len,
                week_window,
                n_series,
                device))
        
        # Multi-Component Fusion
        self.final_conv = nn.Conv2d(
            in_channels = 3,
            out_channels = 1,
            kernel_size = (1, 1)
        )
        
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)
    
    def _cheb_polynomial(self, L_tilde: Tensor) -> List[Tensor]:

        N = L_tilde.shape[0]
        cheb_polynomials = [torch.eye(N).to(L_tilde.device), L_tilde]

        for i in range(2,self. K):
            cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

        return cheb_polynomials
    
    def forward(
        self,
        x: Tensor,
        As: List[Tensor],
        x_day: Tensor,
        x_week: Tensor,
        **kwargs: Any,
    ) -> Tuple[Tensor, None, None]:
        """
        Forward pass.

        Parameters:
            x: input hour features
            As: list of adjacency matrices
            x_day: input day features
            x_week: input week features

        Shape:
            x: (B, P, N, C)
            output: (B, out_len, N)
        """

        # Chebyshev polynomials for Graph convolution
        cheb_polynomials = [i for i in self._cheb_polynomial(As[0])]

        x_list = [x, x_day, x_week]
        num_x = sum(i is not None for i in x_list)

        if num_x != len(self.astgcn):
            raise ValueError("number of submodule not equals to the number of input")
        
        # recent data
        output_h = self.astgcn[0](x.permute(0, 2, 3, 1), cheb_polynomials).unsqueeze(1)
        # daily-periodic data
        output_d = self.astgcn[1](x_day.permute(0, 2, 3, 1), cheb_polynomials).unsqueeze(1)
        # weekly-periodic data
        output_w = self.astgcn[2](x_week.permute(0, 2, 3, 1), cheb_polynomials).unsqueeze(1)

        # Multi-Component Fusion
        output = torch.cat((output_h, output_d, output_w), dim=1)    # (B, 3, N, out_len)
        output = self.final_conv(output).squeeze(1).permute(0 ,2, 1)

        return output, None, None

class _ASTGCN_submodule(nn.Module):
    """
    ASTGCN submodule.

    Parameters:
        num_blocks: number of ASTGCN blocks
        in_channels: dimension of input channel
        K: oreder of chebyshev graph convolution
        num_chev_filter: output dimension of Chebyshev convolution
        num_time_filter: output dimension of ASTGCN block
        num_window: number of hours/days/weeks for lookback window
        out_len: output sequence length
        t_window: time window
        n_series: number of nodes
        device: device
    """
    def __init__(
        self,
        num_blocks: int,
        in_channels: int,
        K: int,
        num_chev_filter: int,
        num_time_filter: int,
        num_window: int,
        out_len: int,
        t_window: int,
        n_series: int,
        device: str,
    ):
        super(_ASTGCN_submodule, self).__init__()

        self.BlockList = nn.ModuleList(
            [_ASTGCN_block(
                in_channels,
                K,
                num_chev_filter,
                num_time_filter,
                num_window,
                n_series,
                t_window,
                device)])

        self.BlockList.extend(
            [_ASTGCN_block(
                num_time_filter,
                K,
                num_chev_filter,
                num_time_filter,
                1,
                n_series,
                t_window//num_window,
                device) for _ in range(num_blocks - 1)])

        self.final_conv = nn.Conv2d(int(t_window / num_window), out_len, kernel_size=(1, num_time_filter))

        self.to(device)

    def forward(
        self,
        x: Tensor,
        cheb_polynomials: List[Tensor]
    ):
        """
        Forward pass.

        Parameters:
            x: input features
            cheb_polynomials: Chebyshev polynomials

        Shape:
            x: (B, N, C, T)
            output: (B, N, out_len)
        """
        for block in self.BlockList:
            x = block(x, cheb_polynomials)

        # (B, N, C', T')->(B, T', N, C')->(B, out_len, N)->(B, N, out_len)
        output = self.final_conv(x.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)

        return output

class _ASTGCN_block(nn.Module):
    """
    ASTGCN block.

    Parameters:
        in_channels: input channels
        K: oreder of chebyshev graph convolution
        num_chev_filter: output channels of Chebyshev convolution
        num_time_filter: output channels of ASTGCN block
        num_window: number of hours/days/weeks for lookback window
        n_series: number of nodes
        t_window: time window
        device: device
    """
    def __init__(
        self,
        in_channels: int,
        K: int,
        num_chev_filter: int,
        num_time_filter: int,
        num_window: int,
        n_series: int,
        t_window: int,
        device: str,
    ):
        super(_ASTGCN_block, self).__init__()

        self.TAt = _Temporal_Attention_layer(in_channels, n_series, t_window, device)
        self.SAt = _Spatial_Attention_layer(in_channels, n_series, t_window, device)
        self.cheb_conv_SAt = _Cheb_Conv_withSAt(K, in_channels, num_chev_filter, device)

        self.time_conv = nn.Conv2d(
            num_chev_filter,
            num_time_filter,
            kernel_size=(1, 3),
            stride=(1, num_window),
            padding=(0, 1))
        
        self.residual_conv = nn.Conv2d(
            in_channels,
            num_time_filter,
            kernel_size=(1, 1),
            stride=(1, num_window))
        
        self.ln = nn.LayerNorm(num_time_filter)

    def forward(
        self,
        x: Tensor,
        cheb_polynomials: List[Tensor]
    ) -> Tensor:
        """
        Forward pass.

        Parameters:
            x: input features
            cheb_polynomials: Chebyshev polynomials
        
        Shape:
            x: (B, N, C, T)
            output: (B, N, C', T')
        """
        B, N, C, T = x.shape

        # Temporal Attention layer
        temporal_At = self.TAt(x)  # (B, T, T)
        x_TAt = torch.matmul(x.reshape(B, -1, T), temporal_At).reshape(B, N, C, T)

        # Spatial Attention layer
        spatial_At = self.SAt(x_TAt)

        # Chebyshev convolution with Spatial Attention
        spatial_gcn = self.cheb_conv_SAt(x, spatial_At, cheb_polynomials)  # (B, N, C', T)

        # Convolution along the time axis
        time_conv_output = self.time_conv(spatial_gcn.permute(0, 2, 1, 3))  # (B, N, C', T)->(B, C'', N, T')

        # Residual shortcut
        x_residual = self.residual_conv(x.permute(0, 2, 1, 3))  # (B, N, C, T)->(B, C'', N, T')

        # (B, C'', N, T')->(B, T', N, C'')->(B, N, C'', T')
        output = self.ln(F.relu(x_residual + time_conv_output).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)

        return output

class _Spatial_Attention_layer(nn.Module):
    """
    Compute spatial attention scores

    Parameters:
        in_channels: input channels
        n_series: number of nodes
        t_window: number of timesteps
        device: device
    """
    def __init__(
        self, 
        in_channels: int, 
        n_series: int, 
        t_window: int,
        device: str,
    ):
        super(_Spatial_Attention_layer, self).__init__()

        self.W1 = nn.Parameter(torch.FloatTensor(t_window).to(device))
        self.W2 = nn.Parameter(torch.FloatTensor(in_channels, t_window).to(device))
        self.W3 = nn.Parameter(torch.FloatTensor(in_channels).to(device))
        self.bs = nn.Parameter(torch.FloatTensor(1, n_series, n_series).to(device))
        self.Vs = nn.Parameter(torch.FloatTensor(n_series, n_series).to(device))

    def forward(
        self, 
        x: Tensor
    ) -> Tensor:
        """
        Forward pass.

        Parameters:
            x: input features
        
        Return:
            S_normalized: normalized spatial attention scores

        Shape:
            x: (B, N, C, T)
            S_normalized: (B, N, N)
        """
        # (B, N, C, T)(T)->(B, N, C)(C, T)->(B, N, T)
        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)

        # (C)(B, N, C, T)->(B, N, T)->(B, T, N)
        rhs = torch.matmul(self.W3, x).transpose(-1, -2)

        # (B,N,T)(B,T,N) -> (B,N,N)
        product = torch.matmul(lhs, rhs)

        # (N, N)(B, N, N)->(B, N, N)
        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))

        S_normalized = F.softmax(S, dim=1)

        return S_normalized

class _Cheb_Conv_withSAt(nn.Module):
    """
    K-order chebyshev graph convolution.

    Parameters:
        K: oreder of chebyshev graph convolution
        in_channles: input channels
        out_channels: output channels
        device: device
    """
    def __init__(
        self,
        K: int, 
        in_channels: int, 
        out_channels: int,
        device: str
    ):
        super(_Cheb_Conv_withSAt, self).__init__()

        self.K = K
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device
        self.Theta = nn.ParameterList(
            [nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(device)) for _ in range(K)])

    def forward(
        self,
        x: Tensor,
        spatial_attention: Tensor,
        cheb_polynomials: List[Tensor]
    ) -> Tensor:
        """
        Forward pass.

        Parameters:
            x: input features.
            spatial_attention: spatial attention scores
            cheb_polynomials: Chebyshev polynomials

        Shape:
            x: (B, N, C, T)
            output: (B, N, C', T)
        """

        B, N, C, T = x.shape

        outputs = []
        for time_step in range(T):
            graph_signal = x[:, :, :, time_step]  # (B, N, C)
            output = torch.zeros(B, N, self.out_channels).to(self.device)  # (B, N, C')
            for k in range(self.K):
                T_k = cheb_polynomials[k]  # (N, N)
                T_k_with_at = T_k.mul(spatial_attention)   # (N, N)(B, N, N)->(B, N, N)
                theta_k = self.Theta[k]  # (in_channel, out_channel)
                rhs = T_k_with_at.permute(0, 2, 1).matmul(graph_signal)  # (B, N, N)(B, N, C)->(B, N, C)
                output = output + rhs.matmul(theta_k)  # (B, N, C)(C, C')->(B, N, C')

            outputs.append(output.unsqueeze(-1))  # (B, N, C', 1)
        
        output = F.relu(torch.cat(outputs, dim=-1))  # (B, N, C', T)

        return output
    
class _Temporal_Attention_layer(nn.Module):
    """
    Compute temporal attention scores

    Parameters:
        in_channels: input channels
        n_series: number of nodes
        t_window: number of timesteps
        device: device
    """
    def __init__(
        self,
        in_channels: int,
        n_series: int,
        t_window: int,
        device: str
    ):
        super(_Temporal_Attention_layer, self).__init__()

        self.U1 = nn.Parameter(torch.FloatTensor(n_series).to(device))
        self.U2 = nn.Parameter(torch.FloatTensor(in_channels, n_series).to(device))
        self.U3 = nn.Parameter(torch.FloatTensor(in_channels).to(device))
        self.be = nn.Parameter(torch.FloatTensor(1, t_window, t_window).to(device))
        self.Ve = nn.Parameter(torch.FloatTensor(t_window, t_window).to(device))

    def forward(
        self,
        x: Tensor
    ) -> Tensor:
        """
        Forward pass.

        Parameters:
            x: input features
        
        Return:
            E_normalized: normalized temporal attention scores

        Shape:
            x: (B, N, C, T)
            E_normalized: (B, T, T)
        """
        # x: (B, N, C, T)->(B, T, C, N)
        # (B, T, C, N)(N)->(B, T, C)(C, N)->(B, T, N)
        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U1), self.U2)

        # (C)(B, N, C, T)->(B, N, T)
        rhs = torch.matmul(self.U3, x)

        # (B, T, N)(B, N, T)->(B, T, T)
        product = torch.matmul(lhs, rhs)

        # (T, T)(B, T, T)->(B, T, T)
        E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))

        E_normalized = F.softmax(E, dim=1)

        return E_normalized