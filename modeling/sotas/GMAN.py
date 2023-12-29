"""
Baseline method, GMAN [AAAI, 2020].
Author: ChunWei Shen

Reference:
* Paper: https://arxiv.org/abs/1911.08415
* Code: 
    * https://github.com/zhengchuanpan/GMAN
    * https://github.com/benedekrozemberczki/pytorch_geometric_temporal
"""
from typing import List, Dict, Any, Optional, Union, Callable, Tuple

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

import numpy as np

class GMAN(nn.Module):
    """
    GMAN.

    Parameters:
        L: number of attention layers
        K: number of attention heads
        d: output dimension of each attention head
        bn_decay: batch normalization momentum
        use_bias: whether to add bias
        t_window: lookback timw window
        n_tids: number of time slots in one day
        device: device
        aux_data: auxiliary data
    """

    def __init__(
        self,
        st_params: Dict[str, Any],
        device: str,
        aux_data: List[np.ndarray],
    ):
        super(GMAN, self).__init__()

        # Network parameters
        self.st_params = st_params
        self.SE = torch.Tensor(aux_data[0]).to(device)
        # hyperparameters of Spatial/Temporal pattern extractor
        L = self.st_params['L']
        K = self.st_params['K']
        d = self.st_params['d']
        bn_decay = self.st_params['bn_decay']  
        use_bias = self.st_params['use_bias']
        self.t_window = st_params['t_window']
        self.n_tids = self.st_params['n_tids']

        D = K * d

        # Spatio Temporal Embedding
        self.st_embedding = _SpatioTemporalEmbedding(
            D, 
            bn_decay, 
            self.n_tids, 
            use_bias
        )

        # Encoder
        self.st_att_encoder = nn.ModuleList(
            [_SpatioTemporalAttention(K, d, bn_decay) for _ in range(L)]
        )
        # Decoder
        self.st_att_decoder = nn.ModuleList(
            [_SpatioTemporalAttention(K, d, bn_decay) for _ in range(L)]
        )
        
        # Transform Attention
        self.transform_attention = _TransformAttention(K, d, bn_decay)

        self.fully_connected_1 = _FullyConnected(
            input_dims = [1, D],
            units = [D, D],
            activations = [F.relu, None],
            bn_decay = bn_decay,
        )
        self.fully_connected_2 = _FullyConnected(
            input_dims = [D, D],
            units = [D, 1],
            activations = [F.relu, None],
            bn_decay = bn_decay,
        )

    def forward(
        self,
        X: Tensor,
        As: Optional[List[Tensor]] = None,
        ycl: Tensor = None,
        **kwargs: Any,
    ) -> Tuple[Tensor, None, None]:
        """
        Forward pass.

        Parameters:
            X: input features
            As: list of adjacency matrices

        Shape:
            X: (B, P, N, C)
            output: (B, out_len, N)

        Return:
            output: output prediction
        """
        X_d = X[:, :, 1, 1] * self.n_tids
        X_w = X[:, :, 1, 2] * 7
        y_d = ycl[:, :, 1, 1] * self.n_tids
        y_w = ycl[:, :, 1, 2] * 7
        X_TE = torch.cat((X_w.unsqueeze(-1), X_d.unsqueeze(-1)), dim = -1)
        y_TE = torch.cat((y_w.unsqueeze(-1), y_d.unsqueeze(-1)), dim = -1)
        TE = torch.cat((X_TE, y_TE), dim = 1).type(torch.int32)
 
        X = X[:, :, :, 0].unsqueeze(-1)   # (B, T, N, C)
        X = self.fully_connected_1(X)

        # Spatio Temporal Embedding
        STE = self.st_embedding(self.SE, TE, self.n_tids)
        STE_his = STE[:, :self.t_window]
        STE_pred = STE[:, self.t_window:]

        # Encoder
        for net in self.st_att_encoder:
            X = net(X, STE_his)

        # Transform Attention
        X = self.transform_attention(X, STE_his, STE_pred)

        # Decoder
        for net in self.st_att_decoder:
            X = net(X, STE_pred)

        output = torch.squeeze(self.fully_connected_2(X), 3)

        return output, None, None

class _Conv2D(nn.Module):
    """
    2D-convolution block.

    Parameters:
        input_dims: input dimension
        output_dims: output dimension
        kernel_size: kernel size of the convolution
        stride: convolution strides
        use_bias: whether to add bias
        activation: activation function
        bn_decay: batch normalization momentum
    """
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        kernel_size: Union[tuple, list],
        stride: Union[tuple, list] = (1, 1),
        use_bias: bool = True,
        activation: Optional[Callable[[torch.FloatTensor], torch.FloatTensor]] = F.relu,
        bn_decay: Optional[float] = None,
    ):
        super(_Conv2D, self).__init__()

        self.activation = activation
        self.conv2d = nn.Conv2d(
            input_dims,
            output_dims,
            kernel_size,
            stride = stride,
            padding = 0,
            bias = use_bias,
        )

        self.batch_norm = nn.BatchNorm2d(output_dims, momentum = bn_decay)

        torch.nn.init.xavier_uniform_(self.conv2d.weight)
        if use_bias: torch.nn.init.zeros_(self.conv2d.bias)

    def forward(
        self,
        X: Tensor
    ) -> Tensor:
        """
        Forward pass.

        Parameters:
            X: input features

        Shape:
            X: (B, T, N, C)
        """
        X = X.permute(0, 3, 2, 1)   # (B, C, N, T)

        X = self.conv2d(X)
        X = self.batch_norm(X)
        if self.activation is not None:
            X = self.activation(X)

        return X.permute(0, 3, 2, 1)    # (B, T, N, C)
    
class _FullyConnected(nn.Module):
    """
    Fully-connected layer.

    Parameters:
        input_dims: input dimension
        units: dimension(s) of outputs in each 2D convolution block
        activations: activation function(s)
        bn_decay: batch normalization momentum
        use_bias: whether to add bias
    """

    def __init__(
        self,
        input_dims: Union[int, list],
        units: Union[int, list],
        activations: Union[Callable[[torch.FloatTensor], torch.FloatTensor], list],
        bn_decay: float,
        use_bias: bool = True,
    ):
        super(_FullyConnected, self).__init__()

        if isinstance(units, int):
            units = [units]
            input_dims = [input_dims]
            activations = [activations]

        assert type(units) == list

        self.conv2ds = nn.ModuleList(
            [
                _Conv2D(
                    input_dims = input_dim,
                    output_dims = num_unit,
                    kernel_size = (1, 1),
                    stride = (1, 1),
                    use_bias = use_bias,
                    activation = activation,
                    bn_decay = bn_decay,
                )
                for input_dim, num_unit, activation in zip(
                    input_dims, units, activations
                )
            ]
        )

    def forward(
        self,
        X: Tensor
    ) -> Tensor:
        """
        Forward pass.

        Parameters:
            X: input features

        Shape:
            X: (B, T, N, C)
        """
        for conv in self.conv2ds:
            X = conv(X)

        return X
    
class _SpatioTemporalEmbedding(nn.Module):
    """
    Spatial-Temporal Embedding block

    Parameters:
        D: output dimension
        bn_decay: batch normalization momentum
        n_tids: number of time slots in one day
        use_bias: whether to add bias in Fully Connected layers
    """

    def __init__(
        self,
        D: int,
        bn_decay: float,
        n_tids: int,
        use_bias: bool = True
    ):
        super(_SpatioTemporalEmbedding, self).__init__()

        self.fully_connected_se = _FullyConnected(
            input_dims = [D, D],
            units = [D, D],
            activations = [F.relu, None],
            bn_decay = bn_decay,
            use_bias = use_bias,
        )

        self.fully_connected_te = _FullyConnected(
            input_dims = [n_tids + 7, D],
            units = [D, D],
            activations = [F.relu, None],
            bn_decay = bn_decay,
            use_bias = use_bias,
        )

    def forward(
        self,
        SE: Tensor,
        TE: Tensor,
        n_tids: int
    ) -> Tensor:
        """
        Forward pass.

        Parameters:
            SE: Spatial embedding
            TE: Temporal embedding
            n_tids: number of time slots in one day

        Shape:
            SE: (N, D)
            TE: (B, t_window + out_len, 2)
            output: (B, t_window + out_len, N ,D)

        Return:
            output: Spatial-Temporal embedding
        """

        SE = SE.unsqueeze(0).unsqueeze(0)   # (1, 1, N, D)
        SE = self.fully_connected_se(SE)

        dayofweek = torch.empty(TE.shape[0], TE.shape[1], 7).to(SE.device)
        timeofday = torch.empty(TE.shape[0], TE.shape[1], n_tids).to(SE.device)
        for i in range(TE.shape[0]):
            dayofweek[i] = F.one_hot(TE[..., 0][i].to(torch.int64) % 7, 7)
        for j in range(TE.shape[0]):
            timeofday[j] = F.one_hot(TE[..., 1][j].to(torch.int64) % n_tids, n_tids)
        TE = torch.cat((dayofweek, timeofday), dim = -1)
        TE = TE.unsqueeze(dim = 2)  # (B, t_window + out_len, 1, n_tids + 7)
        TE = self.fully_connected_te(TE)

        del dayofweek, timeofday

        output = SE + TE

        return output
    
class _SpatialAttention(nn.Module):
    """
    Spatial Attention mechanism.

    Parameters:
        K: number of attention heads
        d: output dimension of each attention head
        bn_decay: batch normalization momentum
    """
    def __init__(
        self,
        K: int,
        d: int,
        bn_decay: float
    ):
        super(_SpatialAttention, self).__init__()

        D = K * d
        self.d = d
        self.K = K

        self.fully_connected_q = _FullyConnected(
            input_dims = 2 * D,
            units = D,
            activations = F.relu,
            bn_decay = bn_decay
        )
        self.fully_connected_k = _FullyConnected(
            input_dims = 2 * D,
            units = D,
            activations = F.relu,
            bn_decay = bn_decay
        )
        self.fully_connected_v = _FullyConnected(
            input_dims = 2 * D,
            units = D,
            activations = F.relu,
            bn_decay = bn_decay
        )
        self.fully_connected = _FullyConnected(
            input_dims = D,
            units = D,
            activations = F.relu,
            bn_decay = bn_decay
        )

    def forward(
        self,
        X: Tensor,
        STE: Tensor
    ) -> Tensor:
        """
        Forward pass.

        Parameters:
            X: input features
            STE: Spatial-Temporal embedding

        Shape:
            X: (B, T, N, K*d)
            STE: (B, T, N, K*d)
            output: (B, T, N, K*d)

        Return:
            output: Spatial attention scores
        """

        batch_size = X.shape[0]
        X = torch.cat((X, STE), dim = -1)   # Concat along channel, (B, T, N, 2*K*d)

        query = self.fully_connected_q(X)   # (B, T, N, K*d)
        key = self.fully_connected_k(X)     # (B, T, N, K*d)
        value = self.fully_connected_v(X)   # (B, T, N, K*d)

        query = torch.cat(torch.split(query, self.K, dim = -1), dim = 0)    # (B*K, T, N, d)
        key = torch.cat(torch.split(key, self.K, dim = -1), dim = 0)        # (B*K, T, N, d)
        value = torch.cat(torch.split(value, self.K, dim = -1), dim = 0)    # (B*K, T, N, d)

        attention = torch.matmul(query, key.transpose(2, 3))    # (B*K, T, N, N)
        attention /= self.d ** 0.5
        attention = F.softmax(attention, dim = -1)

        X = torch.matmul(attention, value)  # (B*K, T, N, d)
        X = torch.cat(torch.split(X, batch_size, dim = 0), dim = -1)    # (B, T, N, K*d)

        output = self.fully_connected(X)
        del query, key, value, attention
        
        return output
    
class _TemporalAttention(nn.Module):
    """
    Temporal attention mechanism.

    Parameters:
        K: number of attention heads
        d: output dimension of each attention head
        bn_decay: batch normalization momentum
        mask: whether to mask attention score
    """
    def __init__(
        self,
        K: int,
        d: int,
        bn_decay: float,
        mask: bool
    ):
        super(_TemporalAttention, self).__init__()

        D = K * d
        self.d = d
        self.K = K
        self.mask = mask

        self.fully_connected_q = _FullyConnected(
            input_dims = 2 * D,
            units = D,
            activations = F.relu,
            bn_decay = bn_decay
        )
        self.fully_connected_k = _FullyConnected(
            input_dims = 2 * D,
            units = D,
            activations = F.relu,
            bn_decay = bn_decay
        )
        self.fully_connected_v = _FullyConnected(
            input_dims = 2 * D,
            units = D,
            activations = F.relu,
            bn_decay = bn_decay
        )
        self.fully_connected = _FullyConnected(
            input_dims = D,
            units = D,
            activations = F.relu,
            bn_decay = bn_decay
        )

    def forward(
        self,
        X: Tensor,
        STE: Tensor
    ) -> Tensor:
        """
        Forward pass.

        Parameters:
            X: input features
            STE: Spatial-Temporal embedding

        Shape:
            X: (B, T, N, K*d)
            STE: (B, T, N, K*d)
            output: (B, T, N, K*d)

        Return:
            output: Temporal attention scores
        """
        batch_size = X.shape[0]
        X = torch.cat((X, STE), dim = -1)   # Concat along channel, (B, T, N, 2*K*d)

        query = self.fully_connected_q(X)
        key = self.fully_connected_k(X)
        value = self.fully_connected_v(X)

        query = torch.cat(torch.split(query, self.K, dim = -1), dim = 0)    # (B*K, T, N, d)
        key = torch.cat(torch.split(key, self.K, dim = -1), dim = 0)        # (B*K, T, N, d)
        value = torch.cat(torch.split(value, self.K, dim = -1), dim = 0)    # (B*K, T, N, d)

        query = query.permute(0, 2, 1, 3)   # (B*K, N, T, d)
        key = key.permute(0, 2, 3, 1)       # (B*K, N, T, d)
        value = value.permute(0, 2, 1, 3)   # (B*K, N, T, d)

        attention = torch.matmul(query, key)    # (B*K, N, T, T)
        attention /= self.d ** 0.5

        if self.mask:
            batch_size = X.shape[0]
            t_window = X.shape[1]
            n_series = X.shape[2]
            mask = torch.ones(t_window, t_window).to(X.device)
            mask = torch.tril(mask)
            mask = torch.unsqueeze(torch.unsqueeze(mask, dim = 0), dim = 0)
            mask = mask.repeat(self.K * batch_size, n_series, 1, 1)
            mask = mask.to(torch.bool)
            condition = torch.FloatTensor([-(2 ** 15) + 1]).to(X.device)
            attention = torch.where(mask, attention, condition)

        attention = F.softmax(attention, dim = -1)
        X = torch.matmul(attention, value)  # (B*K, N, T, d)
        X = X.permute(0, 2, 1, 3)           # (B*K, T, N, d)
        X = torch.cat(torch.split(X, batch_size, dim = 0), dim = -1)    # (B, N, T, K*d)

        output = self.fully_connected(X)    # (B, N, T, K*d)

        del query, key, value, attention

        return output
    
class _GatedFusion(nn.Module):
    """
    Gated fusion mechanism.

    Parameters:
        D: output dimension
        bn_decay: batch normalization momentum
    """

    def __init__(
        self,
        D: int,
        bn_decay: float
    ):
        super(_GatedFusion, self).__init__()

        self.fully_connected_xs = _FullyConnected(
            input_dims = D,
            units = D,
            activations = None,
            bn_decay = bn_decay,
            use_bias = False
        )
        self.fully_connected_xt = _FullyConnected(
            input_dims = D,
            units = D,
            activations = None,
            bn_decay = bn_decay,
            use_bias = True
        )
        self.fully_connected_h = _FullyConnected(
            input_dims = [D, D],
            units = [D, D],
            activations = [F.relu, None],
            bn_decay = bn_decay,
        )

    def forward(
        self,
        HS: Tensor,
        HT: Tensor
    ) -> Tensor:
        """
        Forward pass.

        Parameters:
            HS: Spatial attention scores
            HT: Temporal attention scores
        
        Shape:
            HS: (B, T, N, D)
            HT: (B, T, N, D)
            H: (B, T, N, D)

        Return:
            H: Spatial-Temporal attention scores
        """

        XS = self.fully_connected_xs(HS)
        XT = self.fully_connected_xt(HT)
        z = torch.sigmoid(torch.add(XS, XT))

        H = torch.add(torch.mul(z, HS), torch.mul(1 - z, HT))
        H = self.fully_connected_h(H)

        del XS, XT, z

        return H
    
class _SpatioTemporalAttention(nn.Module):
    """
    Spatial-temporal attention block.

    Parameters:
        K: number of attention heads
        d : output dimension of each attention head
        bn_decay: batch normalization momentum
        mask: whether to mask attention score in temporal attention
    """
    def __init__(
        self,
        K: int,
        d: int,
        bn_decay: float,
        mask: bool = True
    ):
        super(_SpatioTemporalAttention, self).__init__()

        self.spatial_attention = _SpatialAttention(K, d, bn_decay)
        self.temporal_attention = _TemporalAttention(K, d, bn_decay, mask = mask)
        self.gated_fusion = _GatedFusion(K * d, bn_decay)

    def forward(
        self,
        X: Tensor,
        STE: Tensor
    ) -> Tensor:
        """
        Forward pass.

        Parameters:
            X: input features
            STE: Spatial-Temporal embedding
        
        Shape:
            X: (B, T, N, D)
            STE: (B, T, N, D)
            output: (B, T, N, D)

        Return:
            output: attention scores
        """
        HS = self.spatial_attention(X, STE)
        HT = self.temporal_attention(X, STE)
        H = self.gated_fusion(HS, HT)
        del HS, HT

        output = torch.add(X, H)

        return output

class _TransformAttention(nn.Module):
    """
    Tranform attention mechanism.

    Parameters:
        K: number of attention heads.
        d: output dimension of each attention head
        bn_decay: batch normalization momentum
    """
    def __init__(
        self,
        K: int,
        d: int,
        bn_decay: float
    ):
        super(_TransformAttention, self).__init__()

        D = K * d
        self.K = K
        self.d = d

        self.fully_connected_q = _FullyConnected(
            input_dims = D,
            units = D,
            activations = F.relu,
            bn_decay = bn_decay
        )
        self.fully_connected_k = _FullyConnected(
            input_dims = D,
            units = D,
            activations = F.relu,
            bn_decay = bn_decay
        )
        self.fully_connected_v = _FullyConnected(
            input_dims = D,
            units = D,
            activations = F.relu,
            bn_decay = bn_decay
        )
        self.fully_connected = _FullyConnected(
            input_dims = D,
            units = D,
            activations = F.relu,
            bn_decay = bn_decay
        )

    def forward(
        self,
        X: Tensor,
        STE_his: Tensor,
        STE_pred: Tensor,
    ) -> Tensor:
        """
        Forward pass.

        Parameters:
            X: input features
            STE_his: Spatial-Temporal embedding for history
            STE_pred: Spatial-Temporal embedding for prediction
        
        Shape:
            X: (B, T, N, K*d)
            STE_his: (B, T, N, K*d)
            STE_pred: (B, T, N, K*d)
            output: (B, T, N, K*d)

        Return:
            output: output sequence for decoder
        """

        batch_size = X.shape[0]

        query = self.fully_connected_q(STE_pred)    # (B, T, N, K*d)
        key = self.fully_connected_k(STE_his)       # (B, T, N, K*d)
        value = self.fully_connected_v(X)           # (B, T, N, K*d)

        query = torch.cat(torch.split(query, self.K, dim = -1), dim = 0)    # (B*K, T, N, d)
        key = torch.cat(torch.split(key, self.K, dim = -1), dim = 0)        # (B*K, T, N, d)
        value = torch.cat(torch.split(value, self.K, dim = -1), dim = 0)    # (B*K, T, N, d)

        query = query.permute(0, 2, 1, 3)   # (B*K, N, T, d)
        key = key.permute(0, 2, 3, 1)       # (B*K, N, d, T)
        value = value.permute(0, 2, 1, 3)   # (B*K, N, T, d)

        attention = torch.matmul(query, key)    # (B*K, N, T, T)
        attention /= self.d ** 0.5
        attention = F.softmax(attention, dim = -1)

        X = torch.matmul(attention, value)  # (B*K, N, T, d)
        X = X.permute(0, 2, 1, 3)           # (B*K, T, N, d)
        X = torch.cat(torch.split(X, batch_size, dim = 0), dim = -1)    # (B, T, N, K*d)

        output = self.fully_connected(X)    # (B, T, N, K*d)

        del query, key, value, attention

        return output