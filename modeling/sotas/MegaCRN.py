"""
Baseline method, MegaCRN [AAAI, 2023].
Author: ChunWei Shen

Reference:
* Paper: https://arxiv.org/abs/2212.05989
* Code: https://github.com/deepkashiwa20/MegaCRN
"""
import numpy as np
from typing import List, Any, Dict, Tuple, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from torch.nn.modules.loss import _Loss
from utils.scaler import MaxScaler, StandardScaler

class MegaCRN(nn.Module):
    def __init__(
        self,
        st_params: Dict[str, Any],
        mem_params: Dict[str, Any],
        loss_params: Dict[str, Any],
        out_len: int
    ):
        """
        MegaCRN.

        Parameters:
            n_series: number of nodes
            enc_in_dim: input dimension of encoder
            dec_in_dim: input dimension of decoder
            out_dim: dimension of output
            rnn_units: hidden dimension
            num_layers: number of encoder/decoder layers
            cheb_k: order of the Chebyshev polynomials
            cl_decay_steps: control the decay rate of cl threshold
            use_curriculum_learning: if True, model is trained with scheduled sampling
            mem_num: number of memory items
            mem_dim: dimension of each memory item
            out_len: output sequence length
        """
        super(MegaCRN, self).__init__()

        # Network parameters
        self.st_params = st_params
        self.mem_params = mem_params
        self.loss_params = loss_params
        self.out_len = out_len
        # hyperparameters of Spatial/Temporal pattern extractor
        self.n_series = self.st_params['n_series']
        self.enc_in_dim = self.st_params['enc_in_dim']
        self.dec_in_dim = self.st_params['dec_in_dim']
        self.out_dim = self.st_params['out_dim']
        self.rnn_units = self.st_params['rnn_units']
        self.num_layers = self.st_params['num_layers']
        self.cheb_k = self.st_params['cheb_k']
        self.cl_decay_steps = self.st_params['cl_decay_steps']
        self.use_curriculum_learning = self.st_params['use_curriculum_learning']
        # hyperparameters of memory
        self.mem_num = self.mem_params['mem_num']
        self.mem_dim = self.mem_params['mem_dim']
        # hyperparameters of loss
        self.lamb = self.loss_params['lamb']
        self.lamb1 = self.loss_params['lamb1']
        
        # memory
        self.memory = self.construct_memory()

        # encoder
        self.encoder = _Encoder(self.n_series, self.enc_in_dim, self.rnn_units, self.cheb_k, self.num_layers)
        
        # deocoder
        self.decoder_dim = self.rnn_units + self.mem_dim
        self.decoder = _Decoder(self.n_series, self.dec_in_dim, self.decoder_dim, self.cheb_k, self.num_layers)

        # output
        self.proj = nn.Sequential(nn.Linear(self.decoder_dim, self.out_dim, bias=True))
    
    def compute_sampling_threshold(self, batches_seen: int) -> float:
        return self.cl_decay_steps / (self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def construct_memory(self) -> Dict[str, Tensor]:
        memory_dict = nn.ParameterDict()
        memory_dict['Memory'] = nn.Parameter(torch.randn(self.mem_num, self.mem_dim), requires_grad=True)   # (M, d)
        memory_dict['Wq'] = nn.Parameter(torch.randn(self.rnn_units, self.mem_dim), requires_grad=True)     # project to query
        memory_dict['We1'] = nn.Parameter(torch.randn(self.n_series, self.mem_num), requires_grad=True)     # project memory to embedding
        memory_dict['We2'] = nn.Parameter(torch.randn(self.n_series, self.mem_num), requires_grad=True)     # project memory to embedding

        for param in memory_dict.values():
            nn.init.xavier_normal_(param)

        return memory_dict
    
    def query_memory(self, h_t: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        query = torch.matmul(h_t, self.memory['Wq'])     # (B, N, d)
        att_score = torch.softmax(torch.matmul(query, self.memory['Memory'].t()), dim=-1)   # (B, N, M)
        value = torch.matmul(att_score, self.memory['Memory'])     # (B, N, d)
        _, ind = torch.topk(att_score, k=2, dim=-1)
        pos = self.memory['Memory'][ind[:, :, 0]]   # B, N, d
        neg = self.memory['Memory'][ind[:, :, 1]]   # B, N, d

        return value, query, pos, neg
            
    def forward(
        self,
        x: Tensor,
        As: Optional[List[Tensor]] = None,
        ycl: Optional[Tensor] = None,
        iteration: Optional[int] = None,
        **kwargs: Any,
    ) -> Tuple[Tensor, Tensor, Tuple[Tensor, Tensor, Tensor]]:
        """
        Forward pass.

        Parameters:
            input: input features
            As: list of adjacency matrices
            ycl: ground truth
            iteration: number of batches already run

        Shape:
            input: (B, P, N, C)
            ycl: (B, Q, N, C)
            output: (B, Q, N)
        """
        # Spatio-Temporal Meta-Graph Learner
        node_embeddings1 = torch.matmul(self.memory['We1'], self.memory['Memory'])
        node_embeddings2 = torch.matmul(self.memory['We2'], self.memory['Memory'])
        g1 = F.softmax(F.relu(torch.mm(node_embeddings1, node_embeddings2.T)), dim=-1)
        g2 = F.softmax(F.relu(torch.mm(node_embeddings2, node_embeddings1.T)), dim=-1)
        supports = [g1, g2]

        # Encoder
        init_state = self.encoder.init_hidden(x.shape[0])
        h_en, state_en = self.encoder(x[..., :self.enc_in_dim], init_state, supports) # (B, T, N, hid_dim)
        h_t = h_en[:, -1, :, :]                                # (B, N, hid_dim)     
        
        h_att, query, pos, neg = self.query_memory(h_t)
        h_t = torch.cat([h_t, h_att], dim=-1)
        
        # Decoder
        ht_list = [h_t] * self.num_layers
        go = torch.zeros((x.shape[0], self.n_series, self.out_dim), device=x.device)
        out = []
        for t in range(self.out_len):
            h_de, ht_list = self.decoder(torch.cat([go, ycl[:, t, :, 1:]], dim=-1), ht_list, supports)
            go = self.proj(h_de)    # (B, T, out_dim)
            out.append(go)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self.compute_sampling_threshold(iteration):
                    go = ycl[:, t, :, :1]

        output = torch.stack(out, dim=1)    # (B, out_len, N, out_dim)
        output = output.squeeze()           # (B, out_len, N)
        
        return output, h_att, (query, pos, neg)
    
    def train_loss(
        self,
        y_pred: Tensor,
        y_true: Tensor,
        output1: Tensor,
        output2: Union[Tensor, None],
        scaler: Union[MaxScaler, StandardScaler],
        criterion: _Loss,
    ) -> float:
        """
        loss function.

        Parameters:
            y_pred: model prediction
            y_true: ground truth
            output1: h_att
            output2: query, pos, neg
            scaler: scaler
        """
        h_att = output1
        query, pos, neg = output2

        separate_loss = nn.TripletMarginLoss(margin=1.0)
        compact_loss = nn.MSELoss()

        loss1 = criterion(y_pred, y_true)
        loss2 = separate_loss(query, pos.detach(), neg.detach())
        loss3 = compact_loss(query, pos.detach())

        loss = loss1 + self.lamb * loss2 + self.lamb1 * loss3

        return loss
    
    def eval_loss(
        self,
        y_pred: Tensor,
        y_true: Tensor,
        output1: Tensor,
        output2: Union[Tensor, None],
        scaler: Union[MaxScaler, StandardScaler],
        criterion: _Loss,
    ) -> Tuple[float, Tensor, Tensor]:
        """
        loss function.

        Parameters:
            y_pred: model prediction
            y_true: ground truth
            output1: h_att
            output2: query, pos, neg
            scaler: scaler
            criterion: criterion
        """
        y_pred_inv = scaler.inverse_transform(y_pred)
        y_true_inv = scaler.inverse_transform(y_true)
        
        h_att = output1
        query, pos, neg = output2

        separate_loss = nn.TripletMarginLoss(margin=1.0)
        compact_loss = nn.MSELoss()

        loss1 = criterion(y_pred_inv, y_true_inv)
        loss2 = separate_loss(query, pos.detach(), neg.detach())
        loss3 = compact_loss(query, pos.detach())

        loss = loss1 + self.lamb * loss2 + self.lamb1 * loss3

        return loss, y_pred, y_true

class _Decoder(nn.Module):
    def __init__(
        self,
        n_series: int,
        in_dim: int,
        out_dim: int,
        cheb_k: int,
        num_layers: int
    ):
        """
        Decoder.

        Parameters:
            n_series: number of nodes
            in_dim: input dimension
            out_dim: output dimension
            cheb_k: order of the Chebyshev polynomials
            num_layers: number of encoder layers
        """
        super(_Decoder, self).__init__()
        
        self.num_layers = num_layers
        self.n_series = n_series
        self.in_dim = in_dim

        self.dcrnn_cells = nn.ModuleList()
        for layer in range(num_layers):
            in_dim = in_dim if layer == 0 else out_dim
            self.dcrnn_cells.append(_AGCRNCell(n_series, in_dim, out_dim, cheb_k))

    def forward(
        self,
        xt: Tensor, 
        init_state: Tensor, 
        As: List[Tensor]
    ) -> Tensor:
        """
        Forward pass.

        Parameters:
            xt: hidden state of encoder
            init_state: initial state
            As: list of adjacency matrices

        Return:
            current_inputs: output of last layer
            output_hidden: layer-wise hidden state

        Shape:
            xt: (B, N, D)
            init_state: (num_layers, B, N, hid_dim)
            current_inputs: (B, N, hid_dim)
            output_hidden: (num_layers, B, N, hid_dim)
        """
        current_inputs = xt
        output_hidden = []
        for i in range(self.num_layers):
            state = self.dcrnn_cells[i](current_inputs, init_state[i], As)
            output_hidden.append(state)
            current_inputs = state

        return current_inputs, output_hidden

class _Encoder(nn.Module):
    def __init__(
        self, 
        n_series: int, 
        in_dim: int, 
        out_dim: int, 
        cheb_k: int, 
        num_layers: int
    ):
        """
        Encoder.

        Parameters:
            n_series: number of nodes
            in_dim: input dimension
            out_dim: output dimension
            cheb_k: order of the Chebyshev polynomials
            num_layers: number of encoder layers
        """
        super(_Encoder, self).__init__()
        
        self.num_layers = num_layers
        self.n_series = n_series
        self.in_dim = in_dim

        self.dcrnn_cells = nn.ModuleList()
        for layer in range(num_layers):
            in_dim = in_dim if layer == 0 else out_dim
            self.dcrnn_cells.append(_AGCRNCell(n_series, in_dim, out_dim, cheb_k))

    def forward(
        self,
        x: Tensor,
        init_state: Tensor,
        As: List[Tensor]
    ) -> Tensor:
        """
        Forward pass.

        Parameters:
            x: input features
            init_state: initial state
            As: list of adjacency matrices
        
        Return:
            current_inputs: output of last layer
            output_hidden: layer-wise last hidden state
        
        Shape:
            x: (B, T, N, C)
            init_state: (num_layers, B, N, hid_dim)
            current_inputs: (B, T, N, hid_dim)
            output_hidden: (num_layers, B, N, hid_dim)
        """
        t_window = x.shape[1]
        current_inputs = x

        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(t_window):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, As)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
            
        return current_inputs, output_hidden
    
    def init_hidden(
        self,
        batch_size: int
    ) -> Tensor:
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))

        return init_states

class _AGCRNCell(nn.Module):
    def __init__(
        self,
        n_series: int,
        in_dim: int,
        out_dim: int,
        cheb_k: int
    ):
        """
        AGCRNCell.

        Parameters:
            n_series: number of nodes
            in_dim: input dimension
            out_dim: output dimension
            cheb_k: order of the Chebyshev polynomials
        """
        super(_AGCRNCell, self).__init__()

        self.n_series = n_series
        self.hid_dim = out_dim
        self.gate = _AGCN(in_dim + self.hid_dim, 2 * out_dim, cheb_k)
        self.update = _AGCN(in_dim + self.hid_dim, out_dim, cheb_k)

    def forward(
        self,
        x: Tensor,
        state: Tensor,
        As: List[Tensor]
    ) -> Tensor:
        """
        Forward pass.

        Parameters:
            x: input features
            state: hidden state
            As: list of adjacency matrices

        Shape:
            x: (B, N, C)
            state: (B, N, hid_dim)
            h: (B, N, out_dim)
        """
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)     # (B, N, D)

        z_r = torch.sigmoid(self.gate(input_and_state, As)) # (B, N, 2 * out_dim)
        z, r = torch.split(z_r, self.hid_dim, dim=-1)       # (B, N, out_dim), (B, N, out_dim)
        candidate = torch.cat((x, z * state), dim=-1)       # (B, N, D)
        hc = torch.tanh(self.update(candidate, As))         # (B, N, out_dim)
        h = r * state + (1 - r) * hc                        # (B, N, out_dim)

        return h

    def init_hidden_state(
        self,
        batch_size: int
    ) -> Tensor:
        return torch.zeros(batch_size, self.n_series, self.hid_dim)

class _AGCN(nn.Module):
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        cheb_k: int
    ):
        """
        AGCN.

        Parameters:
            in_dim: input dimension
            out_dim: output dimension
            cheb_k: order of the Chebyshev polynomials
        """
        super(_AGCN, self).__init__()

        self.cheb_k = cheb_k
        self.weights = nn.Parameter(torch.FloatTensor(2 * cheb_k * in_dim, out_dim)) # 2 is the length of support
        self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        nn.init.xavier_normal_(self.weights)
        nn.init.constant_(self.bias, val=0)
        
    def forward(
        self,
        x: Tensor,
        As: List[Tensor]
    ) -> Tensor:
        """
        Forward pass.

        Parameters:
            x: input features
            As: list of adjacency matrices

        Shape:
            x: (B, N, C)
            output: (B, N, out_dim)
        """
        
        x_convs = []        
        As_set = []

        for A in As:
            As_cheb = [torch.eye(A.shape[0]).to(A.device), A]
            for k in range(2, self.cheb_k):
                As_cheb.append(torch.matmul(2 * A, As_cheb[-1]) - As_cheb[-2]) 
            As_set.extend(As_cheb)

        for A in As_set:
            x_convs.append(torch.einsum("nm,bmc->bnc", A, x))

        x_convs = torch.cat(x_convs, dim=-1)    # (B, N, 2 * cheb_k * in_dim)
        output = torch.einsum('bni,io->bno', x_convs, self.weights) + self.bias  # (B, N, out_dim)

        return output