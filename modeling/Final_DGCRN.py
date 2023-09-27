"""
DGCRN framework.

Reference: 
https://github.com/tsinghua-fib-lab/Traffic-Benchmark

Author: ChunWei Shen
"""

from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict

class DGCRN(nn.Module):
    """
    DGCRN.

    Parameters:
        st_params: hyperparameters of Spatial/Temporal pattern extractor
        gg_params: hyperparameters of Graph Generator
        out_dim: output dimension
        device: device
        priori_gs: predefined adjacency matrix
    """
    def __init__(
        self,
        st_params: Dict[str, Any],
        gg_params: Dict[str, Any],
        out_dim: int,
        device: str,
        priori_gs: Optional[List[Tensor]] = None,
    ):
        super(DGCRN, self).__init__()

        # Network parameters
        self.st_params = st_params
        self.gg_params = gg_params
        self.predefined_A = [torch.tensor(i).to(device) for i in priori_gs]
        self.out_dim = out_dim
        self.device = device

        # hyperparameters of Spatial/Temporal pattern extractor
        dropout = self.st_params['dropout']
        list_weight = self.st_params['list_weight']
        in_channels = self.st_params["in_channels"]
        gcn_depth = self.st_params['gcn_depth']
        rnn_size = self.st_params['rnn_size']
        self.n_series = self.st_params['n_series']
        self.t_window = self.st_params['t_window']
        self.use_curriculum_learning = self.st_params['use_curriculum_learning']
        self.cl_decay_steps = self.st_params['cl_decay_steps']
        # hyperparameters of Graph Generator
        node_dim = self.gg_params['node_dim']
        middle_dim = self.gg_params['middle_dim']
        hyperGNN_dim = self.gg_params['hyperGNN_dim']
        self.tanhalpha = self.gg_params['tanhalpha']

        self.output_dim = 1
        self.alpha = self.tanhalpha
        self.hidden_size = rnn_size
        self.idx = torch.arange(self.n_series).to(device)
        # node embedding for Graph Generator
        self.embedding1 = nn.Embedding(self.n_series, node_dim)
        self.embedding2 = nn.Embedding(self.n_series, node_dim)

        dims_hyper = [self.hidden_size + in_channels, hyperGNN_dim, middle_dim, node_dim]
        # Graph Generator
        # encoder
        self.GCN1_tg = _GCN(dims_hyper, gcn_depth, dropout, *list_weight, 'hyper')
        self.GCN2_tg = _GCN(dims_hyper, gcn_depth, dropout, *list_weight, 'hyper')
        self.GCN1_tg_1 = _GCN(dims_hyper, gcn_depth, dropout, *list_weight, 'hyper')
        self.GCN2_tg_1 = _GCN(dims_hyper, gcn_depth, dropout, *list_weight, 'hyper')
        # decoder
        self.GCN1_tg_de = _GCN(dims_hyper, gcn_depth, dropout, *list_weight, 'hyper')
        self.GCN2_tg_de = _GCN(dims_hyper, gcn_depth, dropout, *list_weight, 'hyper')
        self.GCN1_tg_de_1 = _GCN(dims_hyper, gcn_depth, dropout, *list_weight, 'hyper')
        self.GCN2_tg_de_1 = _GCN(dims_hyper, gcn_depth, dropout, *list_weight, 'hyper')

        self.fc_final = nn.Linear(self.hidden_size, self.output_dim)

        dims = [self.hidden_size + in_channels, self.hidden_size]
        # DGCRM
        # encoder
        self.gz1 = _GCN(dims, gcn_depth, dropout, *list_weight, 'RNN')
        self.gz2 = _GCN(dims, gcn_depth, dropout, *list_weight, 'RNN')
        self.gr1 = _GCN(dims, gcn_depth, dropout, *list_weight, 'RNN')
        self.gr2 = _GCN(dims, gcn_depth, dropout, *list_weight, 'RNN')
        self.gc1 = _GCN(dims, gcn_depth, dropout, *list_weight, 'RNN')
        self.gc2 = _GCN(dims, gcn_depth, dropout, *list_weight, 'RNN')
        # decoder
        self.gz1_de = _GCN(dims, gcn_depth, dropout, *list_weight, 'RNN')
        self.gz2_de = _GCN(dims, gcn_depth, dropout, *list_weight, 'RNN')
        self.gr1_de = _GCN(dims, gcn_depth, dropout, *list_weight, 'RNN')
        self.gr2_de = _GCN(dims, gcn_depth, dropout, *list_weight, 'RNN')
        self.gc1_de = _GCN(dims, gcn_depth, dropout, *list_weight, 'RNN')
        self.gc2_de = _GCN(dims, gcn_depth, dropout, *list_weight, 'RNN')

    def preprocessing(
        self,
        adj: Tensor,
        predefined_A: Tensor
    ) -> List[Tensor]:
        """
        preprocess the adjacency matrices.

        Parameters:
            adj: dynamic adjacency matrix
            predefined_A: predefined adjacency matrix
        """
        adj = adj + torch.eye(self.n_series).to(self.device)
        adj = adj / torch.unsqueeze(adj.sum(-1), -1)

        return [adj, predefined_A]
    
    def step(
        self,
        input: Tensor,
        Hidden_State: Tensor,
        Cell_State: Tensor,
        predefined_A: List[Tensor],
        type: str = 'encoder',
        idx: Tensor = None,
        i: int = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Step of encoder/decoder.

        Parameters:
            input: input features
            Hidden_State: Hidden state
            Cell_State: Cell state
            predefined_A: predefined adjacency matrices
            type: encoder or decoder
        
        Return:
            Hidden_State: Hidden state
            Cell_State: Cell state
        
        Shape:
            x: (B, C, N)
            Hidden_State: (B*N, hidden_size)
            Cell_State: (B*N, hidden_size)
        """
        x = input
        x = x.transpose(1, 2).contiguous()  #(B, N, C)

        # Graph Generator
        nodevec1 = self.embedding1(self.idx)
        nodevec2 = self.embedding1(self.idx)

        hyper_input = torch.cat(
            (x, Hidden_State.view(-1, self.n_series, self.hidden_size)), 2)

        if type == 'encoder':
            filter1 = (self.GCN1_tg(hyper_input, predefined_A[0]) + 
                       self.GCN1_tg_1(hyper_input, predefined_A[1]))
            filter2 = (self.GCN2_tg(hyper_input, predefined_A[0]) + 
                       self.GCN2_tg_1(hyper_input, predefined_A[1]))
            
        if type == 'decoder':
            filter1 = (self.GCN1_tg_de(hyper_input, predefined_A[0]) + 
                       self.GCN1_tg_de_1(hyper_input, predefined_A[1]))
            filter2 = (self.GCN2_tg_de(hyper_input, predefined_A[0]) + 
                       self.GCN2_tg_de_1(hyper_input, predefined_A[1]))

        nodevec1 = torch.tanh(self.alpha * torch.mul(nodevec1, filter1))
        nodevec2 = torch.tanh(self.alpha * torch.mul(nodevec2, filter2))

        a = (torch.matmul(nodevec1, nodevec2.transpose(2, 1)) 
             - torch.matmul(nodevec2, nodevec1.transpose(2, 1)))

        adj = F.relu(torch.tanh(self.alpha * a))

        adp = self.preprocessing(adj, predefined_A[0])
        adpT = self.preprocessing(adj.transpose(1, 2), predefined_A[1])

        # DGCRM
        Hidden_State = Hidden_State.view(-1, self.n_series, self.hidden_size)
        Cell_State = Cell_State.view(-1, self.n_series, self.hidden_size)
        combined = torch.cat((x, Hidden_State), -1)

        if type == 'encoder':
            z = torch.sigmoid(self.gz1(combined, adp) + self.gz2(combined, adpT))
            r = torch.sigmoid(self.gr1(combined, adp) + self.gr2(combined, adpT))
            temp = torch.cat((x, torch.mul(r, Hidden_State)), -1)
            Cell_State = torch.tanh(self.gc1(temp, adp) + self.gc2(temp, adpT))
        elif type == 'decoder':
            z = torch.sigmoid(self.gz1_de(combined, adp) + self.gz2_de(combined, adpT))
            r = torch.sigmoid(self.gr1_de(combined, adp) + self.gr2_de(combined, adpT))
            temp = torch.cat((x, torch.mul(r, Hidden_State)), -1)
            Cell_State = torch.tanh(self.gc1_de(temp, adp) + self.gc2_de(temp, adpT))

        Hidden_State = torch.mul(z, Hidden_State) + torch.mul((1 - z), Cell_State)

        return Hidden_State.view(-1, self.hidden_size), Cell_State.view(-1, self.hidden_size)
    
    def forward(
        self,
        input: Tensor,
        ycl: Tensor,
        batches_seen: int = None,
        task_level: int = 12,
        idx = None,
        **kwargs: Any,
    ) -> Tuple[Tensor, None, None]:
        """
        Forward pass.

        Parameters:
            input: input features
            ycl: normalized ground truth
            batches_seen: number of batches already run
            task_level: task level for curriculum learning
            idx: index

        Return:
            outputs_final: prediction
        
        Shape:
            input: (B, T, N, C), where B is the batch_size, T is the lookback
                   time window and N is the number of time series
            ycl: (B, Q, N, C)
            output: (B, out_dim, N)
        """
        input = input.permute(0, 3, 2, 1)   # (B, C, N, T)
        ycl = ycl.permute(0, 3, 2, 1)       # (B, C, N, Q)

        if task_level == None: task_level = 12

        x = input
        batch_size = x.size(0)
        predefined_A = self.predefined_A
        
        Hidden_State, Cell_State = self._initHidden(
            batch_size * self.n_series,
            self.hidden_size)

        # Encoder
        outputs = None
        for i in range(self.t_window):
            Hidden_State, Cell_State = self.step(
                torch.squeeze(x[..., i]),   # (B, C, N)
                Hidden_State, 
                Cell_State,
                predefined_A, 
                'encoder', 
                idx,
                i)

            if outputs is None:
                outputs = Hidden_State.unsqueeze(1)   # (B, 1, C, N)
            else:
                outputs = torch.cat((outputs, Hidden_State.unsqueeze(1)), 1)    # (B, T, C, N)

        go_symbol = torch.zeros(
            (batch_size, self.output_dim, self.n_series), 
            device=self.device)
        
        timeofday = ycl[:, 1:, :, :]    # (B, 1, N, Q)
        decoder_input = go_symbol
        outputs_final = []

        # Decoder
        for i in range(task_level):
            try:
                decoder_input = torch.cat([decoder_input, timeofday[..., i]], dim=1)
            except:
                print(decoder_input.shape, timeofday.shape)
            
            Hidden_State, Cell_State = self.step(
                decoder_input, 
                Hidden_State,
                Cell_State, 
                predefined_A,
                'decoder', 
                idx, 
                i=None)

            decoder_output = self.fc_final(Hidden_State)    # (B*N, 1)

            decoder_input = decoder_output.view(
                batch_size, 
                self.n_series,
                self.output_dim).transpose(1, 2)    # (B, N, 1)
            
            outputs_final.append(decoder_output)

            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    decoder_input = ycl[:, :1, :, i]

        outputs_final = torch.stack(outputs_final, dim=1)

        outputs_final = outputs_final.view(
            batch_size, 
            self.n_series,
            task_level).transpose(1, 2)

        return outputs_final, None, None
    
    def _initHidden(self, batch_size, hidden_size):
        '''Initialization of hidden state.'''
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            Hidden_State = Variable(torch.zeros(batch_size, hidden_size).to(self.device))
            Cell_State = Variable(torch.zeros(batch_size, hidden_size).to(self.device))
            nn.init.orthogonal_(Hidden_State)
            nn.init.orthogonal_(Cell_State)

            return Hidden_State, Cell_State
        else:
            Hidden_State = Variable(torch.zeros(batch_size, hidden_size))
            Cell_State = Variable(torch.zeros(batch_size, hidden_size))

            return Hidden_State, Cell_State

    def _compute_sampling_threshold(
        self, 
        batches_seen: int
    ) -> float:
        """Compute scheduled sampling threshold."""
        return (self.cl_decay_steps 
                / (self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps)))

class _GCN(nn.Module):
    """
    Graph Convolution Layer.

    Parameters:
        dims: dimension of input/output
        gcn_depth: depth of graph convolution
        dropout:  dropout ratio
        alpha: alpha
        beta: beta
        gamma: gamma
        type: hyper_network or RNN
    """
    def __init__(
        self,
        dims: List[int], 
        gcn_depth: int, 
        dropout: float, 
        alpha: float, 
        beta: float, 
        gamma: float, 
        type: str = None
    ):
        super(_GCN, self).__init__()

        self.gcn_depth = gcn_depth
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.type_GNN = type

        if type == 'RNN':
            self.mlp = nn.Linear((gcn_depth + 1) * dims[0], dims[1])
        elif type == 'hyper':
            self.mlp = nn.Sequential(
                OrderedDict(
                    [('fc1', nn.Linear((gcn_depth + 1) * dims[0], dims[1])),
                     ('sigmoid1', nn.Sigmoid()),
                     ('fc2', nn.Linear(dims[1], dims[2])),
                     ('sigmoid2', nn.Sigmoid()),
                     ('fc3', nn.Linear(dims[2], dims[3]))]))
        
    def forward(
        self,
        x: Tensor,
        adj: List[Tensor]
    ) -> Tensor:
        """
        Forward pass.

        Parameters:
            x: input features
            adj: adjacency matrices

        Shape:
            x: (B, N, C)
            ho: (B, N, C')
        """

        h = x
        out = [h]

        # GNN for RNN
        if self.type_GNN == "RNN":
            for _ in range(self.gcn_depth):
                h = (self.alpha * x 
                     + self.beta * torch.einsum('nvc,nvw->nwc', (h, adj[0]))
                     + self.gamma * torch.einsum('nvc,vw->nwc', (h, adj[1])))
                out.append(h)
        # GNN for hyper
        else:
            for _ in range(self.gcn_depth):
                h = (self.alpha * x + 
                    self.gamma * torch.einsum('nvc,vw->nwc', (h, adj)))
                out.append(h)

        ho = torch.cat(out, dim = -1)
        ho = self.mlp(ho)

        return ho