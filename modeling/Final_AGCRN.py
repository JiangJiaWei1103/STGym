"""
AGCRN framework.

Reference: 
https://github.com/LeiBAI/AGCRN

Author: ChunWei Shen
"""
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

class AGCRN(nn.Module):
    """
    AGCRN.

    Parameters:
        st_params: hyperparameters of Spatial/Temporal pattern extractor
        dagg_params: hyperparameters of Data Adaptive Graph Generation
        out_dim: output dimension
        priori_gs: predefined adjacency matrix
    """
    def __init__(
        self,
        st_params: Dict[str, Any],
        dagg_params: Dict[str, Any],
        out_dim: int,
        priori_gs: Optional[List[Tensor]] = None,
    ):
        super(AGCRN, self).__init__()

        # Network parameters
        self.st_params = st_params
        self.dagg_params = dagg_params
        self.out_dim = out_dim

        # hyperparameters of Spatial/Temporal pattern extractor
        num_layers = self.st_params['num_layers']
        rnn_units = self.st_params['rnn_units']
        cheb_k = self.st_params['cheb_k']
        in_channels = self.st_params['in_channels']
        self.out_channels = self.st_params['out_channels']
        self.n_series = self.st_params['n_series']
        # hyperparameters of Data Adaptive Graph Generation
        self.embedding_dim = self.dagg_params['embedding_dim']

        self.hidden_dim = rnn_units

        self.node_embeddings = nn.Parameter(
            torch.randn(self.n_series, self.embedding_dim),
            requires_grad = True)

        self.encoder = _AVWDCRNN(
            self.n_series, 
            in_channels, 
            rnn_units, 
            cheb_k,
            self.embedding_dim,
            num_layers)

        # predictor
        self.end_conv = nn.Conv2d(
            1, 
            self.out_dim * self.out_channels, 
            kernel_size = (1, self.hidden_dim), 
            bias=True)
        
        self._reset_parameters()
         
    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(
        self, 
        source: Tensor, 
        teacher_forcing_ratio: float = 0.5,
        **kwargs: Any,
    ) -> Tuple[Tuple, None, None]:
        """
        Forward pass.

        Parameters:
            source: input features
        
        Return:
            output: prediction
        
        Shape:
            source: (B, T, N, C), where B is the batch_size, T is the lookback
                    time window and N is the number of time series
            tid: (B, )
            diw: (B, )
            output: (B, out_dim, N)
        """
        batch_size = source.shape[0]
        init_state = self.encoder.init_hidden(batch_size)

        output, _ = self.encoder(source, init_state, self.node_embeddings)   # (B, T, N, D)
        output = output[:, -1:, :, :]                                        # (B, 1, N, D)

        # CNN based predictor
        output = self.end_conv(output)       # (B, out_dim*C, N, 1)
        output = output.squeeze(-1).reshape(-1, self.out_dim, self.out_channels, self.n_series)
        output = output.permute(0, 1, 3, 2).squeeze(3)   # (B, T, N)

        return output, None, None

class _AVWDCRNN(nn.Module):
    """
    RNN layers.

    Parameters:
        n_series: number of nodes
        dim_in: dimension of input features
        dim_out: dimension of output features
        cheb_k: order of chebyshev polynomial expansion
        embedding_dim: dimension of node embedding
        num_layers: number of layers
    """
    def __init__(
        self, 
        n_series: int, 
        dim_in: int, 
        dim_out: int, 
        cheb_k: int, 
        embedding_dim: int, 
        num_layers: int = 1
    ):
        super(_AVWDCRNN, self).__init__()

        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'

        self.n_series = n_series
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()

        self.dcrnn_cells.append(_AGCRNCell(n_series, dim_in, dim_out, cheb_k, embedding_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(_AGCRNCell(n_series, dim_out, dim_out, cheb_k, embedding_dim))

    def forward(
        self, 
        x: Tensor, 
        init_state: Tensor, 
        node_embeddings: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass.

        Parameters:
            x: input features
            init_state: initialized hidden state
            node_embeddings: node embeddings
        
        Return:
            current_inputs: the outputs of last layer
            output_hidden: the last state for each layer
        
        Shape:
            x: (B, T, N, C)
            init_state: (num_layers, B, N, D)
            current_inputs: (B, T, N, D)
            output_hidden: (num_layers, B, N, D)
        """
  
        assert x.shape[2] == self.n_series and x.shape[3] == self.input_dim

        t_window = x.shape[1]
        current_inputs = x
        output_hidden = []

        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(t_window):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim = 1)

        return current_inputs, output_hidden

    def init_hidden(
        self, 
        batch_size: int
    ) -> Tensor:
        '''Initialization of hidden state.'''
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim = 0)      #(num_layers, B, N, D)

class _AGCRNCell(nn.Module):
    """
    Adaptive Graph Convolutional Recurrent Cell.

    Parameters:
        n_series: number of nodes
        dim_in: dimension of input features
        dim_out: dimension of output features
        cheb_k: order of chebyshev polynomial expansion
        embedding_dim: dimension of node embedding

    """
    def __init__(
        self, 
        n_series: int, 
        dim_in: int, 
        dim_out: int, 
        cheb_k: int, 
        embedding_dim: int
    ):
        super(_AGCRNCell, self).__init__()

        self.n_series = n_series
        self.hidden_dim = dim_out

        self.gate = _AVWGCN(
            dim_in + self.hidden_dim, 
            2 * dim_out, 
            cheb_k, 
            embedding_dim)
        self.update = _AVWGCN(
            dim_in + self.hidden_dim, 
            dim_out, 
            cheb_k, 
            embedding_dim)

    def forward(
        self, 
        x: Tensor, 
        state: Tensor, 
        node_embeddings: Tensor
    ) -> Tensor:
        """
        Forward pass.

        Parameters:
            x: input features
            state: hidden state
            node_embeddings: node embeddings
        
        Return:
            h: current satae

        Shape:
            x: (B, N, C)
            state: (B, N, D)
            h: (B, N, D)
        """

        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim = -1)   # Concat along channel

        # GRU
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim = -1)
        candidate = torch.cat((x, (z * state)), dim = -1)
        hc = torch.tanh(self.update(candidate, node_embeddings))
        h = r * state + (1 - r) * hc

        return h
    
    def init_hidden_state(
        self, 
        batch_size: int
    ) -> Tensor:
        '''
        Initialization of hidden state.
        '''
        return torch.zeros(batch_size, self.n_series, self.hidden_dim)

class _AVWGCN(nn.Module):
    """
    Node Adaptive Parameter Learning-GCN (NAPL-GCN).

    Parameters:
        dim_in: dimension of input features
        dim_out: dimension of output features
        cheb_k: order of chebyshev polynomial expansion
        embedding_dim: dimension of node embedding
    """
    def __init__(
        self, 
        dim_in: int, 
        dim_out: int, 
        cheb_k: int, 
        embedding_dim: int
    ):
        super(_AVWGCN, self).__init__()

        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embedding_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embedding_dim, dim_out))

    def forward(
        self, 
        x: Tensor, 
        node_embeddings: Tensor
    ) -> Tensor:
        """
        Forward pass.

        Parameters:
            x: input features
            node_embeddings: node embeddings

        Return:
            x_gconv: hidden state for all nodes
        
        Shape:
            x: (B, N, C)
            node_embeddings: (N, D)
            x_gconv: (B, N, C')
        """

        n_series = node_embeddings.shape[0]

        # Data Adaptive Graph Generation
        # supports: (N, N)
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim = 1)
        support_set = [torch.eye(n_series).to(supports.device), supports]

        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])

        supports = torch.stack(support_set, dim = 0)

        #  NAPL-GCN
        # (N, cheb_k, dim_in, dim_out)
        weights = torch.einsum('nd,dkio->nkio', (node_embeddings, self.weights_pool))
        # (N, dim_out)
        bias = torch.matmul(node_embeddings, self.bias_pool)

        # (B, cheb_k, N, dim_in)
        x_g = torch.einsum("knm,bmc->bknc", supports, x).permute(0, 2, 1, 3)
        # (B, N, dim_out)
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias

        return x_gconv