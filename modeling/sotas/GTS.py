"""
Baseline method, GTS [ICLR, 2021].
Author: ChunWei Shen

Reference:
* Paper: https://arxiv.org/abs/2101.06861
* Code: https://github.com/chaoshangcs/GTS
"""
from typing import List, Any, Dict, Optional, Callable, Tuple

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

import random
import numpy as np
from utils.scaler import StandardScaler
from sklearn.neighbors import kneighbors_graph
from metadata import TrafBerks

class GTS(nn.Module):
    """
    GTS.

    Parameters:
        num_layers: number of encoder/decoder layers
        rnn_units: hidden dimension of rnn
        out_dim: dimension of output
        encoder_input_dim: input dimension of encoder
        decoder_input_dim: input dimension of decoder
        max_diffusion_step: number of diffusion steps
        n_series: number of nodes
        temperature: non-negative scalar for gumbel softmax
        cl_decay_steps: control the decay rate of cl threshold
        use_curriculum_learning: if True, model is trained with scheduled sampling
        train_ratio: ratio of training data
        k: number of neighbors for kneighbors_graph
        batch_size: batch size
        device: device
        out_len: output sequence length
        aux_data: auxiliary data
    """
    def __init__(
        self, 
        st_params: Dict[str, Any],
        nf_params: Dict[str, Any],
        batch_size: int,
        device: str,
        out_len: int,
        aux_data: List[np.ndarray],
    ):
        super(GTS, self).__init__()

        # Network parameters
        self.st_params = st_params
        self.nf_params = nf_params
        self.out_len = out_len
        self.device = device
        self.batch_size = batch_size
        self.aux_data = aux_data[0]
        # hyperparameters of Spatial/Temporal pattern extractor
        num_layers = self.st_params['num_layers']
        rnn_units = self.st_params['rnn_units']
        out_dim = self.st_params['out_dim']
        encoder_input_dim = self.st_params['encoder_input_dim']
        decoder_input_dim = self.st_params['decoder_input_dim']
        max_diffusion_step = self.st_params['max_diffusion_step']
        self.n_series = self.st_params['n_series']
        self.temp = self.st_params['temperature']
        self.cl_decay_steps = self.st_params['cl_decay_steps']
        self.use_curriculum_learning = self.st_params['use_curriculum_learning']
        # hyperparameters of Node features
        train_ratio = self.nf_params['train_ratio']
        self.dataset_name = self.nf_params['dataset_name']
        k = self.nf_params['k']
        self.node_features = self._node_features_preprocess(
            train_ratio
        ).to(device)

        # Encoder
        self.encoder = _Encoder(
            input_dim=encoder_input_dim,
            max_diffusion_step=max_diffusion_step,
            hid_dim=rnn_units, 
            n_series=self.n_series,
            num_layers=num_layers
        )
        
        # Decoder
        self.decoder = _Decoder(
            input_dim=decoder_input_dim,
            max_diffusion_step=max_diffusion_step,
            n_series=self.n_series,
            hid_dim=rnn_units,
            output_dim=out_dim,
            num_layers=num_layers
        )
        
        assert self.encoder.hid_dim == self.decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        
        # Graph Structure Parameterization
        kernel_size = 10
        conv_hid_dim = 8
        conv_out_dim = 16
        self.embedding_dim = 100
        self.conv1 = torch.nn.Conv1d(
            in_channels=1,
            out_channels=conv_hid_dim,
            kernel_size=kernel_size,
            stride=1
        )
        self.conv2 = torch.nn.Conv1d(
            in_channels=conv_hid_dim,
            out_channels=conv_out_dim,
            kernel_size=kernel_size, 
            stride=1
        )
        
        dim_fc = ((self.node_features.shape[0] - 2 * (kernel_size - 1)) * conv_out_dim)
        self.fc = torch.nn.Linear(dim_fc, self.embedding_dim)
        self.bn1 = torch.nn.BatchNorm1d(8)
        self.bn2 = torch.nn.BatchNorm1d(16)
        self.bn3 = torch.nn.BatchNorm1d(self.embedding_dim)
        self.fc_out = nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        self.fc_cat = nn.Linear(self.embedding_dim, 2)
        self.hidden_drop = torch.nn.Dropout(0.2)

        # Generate off-diagonal interaction graph
        off_diag = np.ones([self.n_series, self.n_series])
        rel_rec = np.array(self._encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        rel_send = np.array(self._encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
        self.rel_rec = torch.FloatTensor(rel_rec).to(device)
        self.rel_send = torch.FloatTensor(rel_send).to(device)

        g = kneighbors_graph(self.node_features.cpu().T, k, metric='cosine')
        g = np.array(g.todense(), dtype=np.float32)
        self.prior_adj = torch.Tensor(g)

    def forward(
        self, 
        input: Tensor, 
        As: Optional[List[Tensor]] = None,
        ycl: Tensor = None,
        iteration: int = None,
        **kwargs: Any,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        
        """
        Forward pass.

        Parameters:
            input: input features
            ycl: normalized ground truth
            batches_seen: number of batches already run

        Shape:
            input: (B, P, N, C)
            ycl: (B, Q, N, C)
            output: (B, out_len, N)
        """
        # Graph Structure Parameterization
        x = self.node_features.transpose(1, 0).view(self.n_series, 1, -1)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = x.view(self.n_series, -1)
        x = self.fc(x)
        x = F.relu(x)
        x = self.bn3(x)

        receivers = torch.matmul(self.rel_rec, x)
        senders = torch.matmul(self.rel_send, x)
        x = torch.cat([senders, receivers], dim=1)
        x = torch.relu(self.fc_out(x))
        x = self.fc_cat(x)
        self.mid_output = x.softmax(-1)[:, 0].clone().reshape(self.n_series, -1)

        adj = self._gumbel_softmax(x, temperature=self.temp, hard=True)
        adj = adj[:, 0].clone().reshape(self.n_series, -1)
        mask = torch.eye(self.n_series, self.n_series).bool().to(self.device)
        adj.masked_fill_(mask, 0)

        input = input.permute(1, 0, 2, 3)   # (T, B, N, C)
        batch_size = input.shape[1]
        init_state = self.encoder.init_hidden(batch_size)
        # Encoder
        output_hidden, _ = self.encoder(input, init_state, adj)  # (num_layers, B, N*D)

        if self.training and self.use_curriculum_learning:
            teacher_forcing_ratio = self._compute_sampling_threshold(iteration)
        else:
            teacher_forcing_ratio = 0

        # Decoder
        outputs = self.decoder(
            ycl, 
            output_hidden, 
            adj, 
            teacher_forcing_ratio=teacher_forcing_ratio
        )
        
        outputs = outputs.permute(1, 0, 2)     # (B, out_len, N)

        return outputs, self.mid_output, self.prior_adj
    
    def _node_features_preprocess(
        self,
        train_ratio: float
    ) -> Tensor:
        """
        Node features preprocess.

        Parameters:
            train_ratio: train ratio

        Return:
            train_features: train features
        """
        try:
            data_vals = self.aux_data["data"][..., 0]
        except:
            data_vals = self.aux_data
        num_samples = data_vals.shape[0]
        num_train = round(num_samples * train_ratio)
        data_vals = data_vals[:num_train]
        scaler = StandardScaler()
        train_features = scaler.fit_transform(data_vals)
        train_features = torch.Tensor(train_features)

        return train_features
    
    def _encode_onehot(
        self, 
        labels: np.array
    ) -> np.array:
        
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                        enumerate(classes)}
        labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                 dtype=np.int32)
        return labels_onehot
    
    def _sample_gumbel(
        self, 
        shape: torch.Size, 
        eps: float = 1e-20
    ) -> Tensor:
        
        U = torch.rand(shape).to(self.device)
        return -torch.autograd.Variable(torch.log(-torch.log(U + eps) + eps))

    def _gumbel_softmax_sample(
        self, 
        logits: Tensor, 
        temperature: int, 
        eps: float = 1e-10
    ) -> Tensor:
        
        sample = self._sample_gumbel(logits.size(), eps=eps)
        y = logits + sample
        return F.softmax(y / temperature, dim=-1)

    def _gumbel_softmax(
        self,
        logits: Tensor, 
        temperature: int, 
        hard: bool = False, 
        eps: float = 1e-10
    ) -> Tensor:
        """
        Sample from the Gumbel-Softmax distribution and optionally discretize.

        Parameters:
            logits: unnormalized log-probs
            temperature: non-negative scalar
            hard: if True, take argmax, but differentiate w.r.t. soft sample y

        Return:
            sample from the Gumbel-Softmax distribution.
            If hard=True, then the returned sample will be one-hot, 
            otherwise it will be a probabilitiy distribution that 
            sums to 1 across classes.
        """
        y_soft = self._gumbel_softmax_sample(logits, temperature=temperature, eps=eps)

        if hard:
            shape = logits.size()
            _, k = y_soft.data.max(-1)
            y_hard = torch.zeros(*shape).to(self.device)
            y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
            y = torch.autograd.Variable(y_hard - y_soft.data) + y_soft
        else:
            y = y_soft

        return y
        
    def _compute_sampling_threshold(
        self, 
        iteration: int
    ) -> float:
        """
        Compute scheduled sampling threshold.
        """
        return (self.cl_decay_steps 
                / (self.cl_decay_steps + np.exp(iteration / self.cl_decay_steps)))

class _Encoder(nn.Module):
    """
    DCRNN Encoder.

    Parameters:
        input_dim: dimension of input features
        hid_dim: hidden dimension of rnn
        num_layers: number of rnn layers
        max_diffusion_step: max diffusion step
        n_series: number of nodes
    """
    def __init__(
        self,
        input_dim: int,
        hid_dim: int,
        num_layers: int,
        max_diffusion_step: int,
        n_series: int,
    ):
        super(_Encoder, self).__init__()

        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.encoder = nn.ModuleList()

        for layer in range(num_layers):
            input_dim = input_dim if layer == 0 else hid_dim
            self.encoder.append(
                _DCGRUCell(
                    input_dim=input_dim,
                    hid_dim=hid_dim,
                    max_diffusion_step=max_diffusion_step,
                    n_series=n_series
                )
            )

    def forward(
        self, 
        inputs: Tensor, 
        init_state: Tensor,
        adj: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass.

        Parameters:
            inputs: input features
            init_state: initialized hidden state
            adj: adjacency matrix

        Return:
            output_hidden: the last state for each layer
            current_inputs: the outputs of last layer
        
        Shape:
            inputs: (T, B, N, C)
            init_state: (num_layers, B, N*D)
            output_hidden: (num_layers, B, N*D)
            current_inputs: (T, B, N*D)
        """

        t_window = inputs.shape[0]
        batch_size = inputs.shape[1]
        inputs = inputs.reshape(t_window, batch_size, -1)  # (T, B, N*C)

        current_inputs = inputs
        output_hidden = []

        # Encoder
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(t_window):
                _, state = self.encoder[i](current_inputs[t, ...], state, adj)    # (B, N*D)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=0)  # (T, B, N*D)

        return output_hidden, current_inputs

    def init_hidden(
        self, 
        batch_size: int
    ) -> Tensor:
        '''Initialization of hidden state.'''
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.encoder[i].init_hidden_state(batch_size))

        return torch.stack(init_states, dim=0)  # (num_layers, B, N*D)

class _Decoder(nn.Module):
    """
    DCRNN Decoder.

    Parameters:
        input_dim: dimension of input features
        output_dim: dimension of output
        hid_dim: hidden dimension of rnn
        num_layers: number of rnn layers
        max_diffusion_step: max diffusion step
        n_series: number of nodes
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hid_dim: int,
        num_layers: int,
        max_diffusion_step: int,
        n_series: int,
    ):
        super(_Decoder, self).__init__()

        self.hid_dim = hid_dim
        self.n_series = n_series
        self.output_dim = output_dim 
        self.num_layers = num_layers
        self.decoder = nn.ModuleList()

        cell = _DCGRUCell(
            input_dim=hid_dim,
            hid_dim=hid_dim,
            max_diffusion_step=max_diffusion_step,
            n_series=n_series
        )
        
        cell_final = _DCGRUCell(
            input_dim=hid_dim,
            hid_dim=hid_dim,
            max_diffusion_step=max_diffusion_step,
            n_series=n_series,
            out_dim=output_dim
        )
        
        if num_layers == 1:
            self.decoder.append( 
                _DCGRUCell(
                    input_dim=input_dim,
                    hid_dim=hid_dim,
                    max_diffusion_step=max_diffusion_step,
                    n_series=n_series,
                    out_dim=output_dim
                )
            )
        else:
            # first layer
            self.decoder.append(
                _DCGRUCell(
                    input_dim=input_dim,
                    hid_dim=hid_dim,
                    max_diffusion_step=max_diffusion_step,
                    n_series=n_series
                )
            )
            
            # multi-layer rnn if num_layers > 1
            for _ in range(1, num_layers - 1):
                self.decoder.append(cell)

            # last layer
            self.decoder.append(cell_final)
        
    def forward(
        self, 
        inputs: Tensor, 
        init_state: Tensor,
        adj: Tensor,
        teacher_forcing_ratio: float = 0.5
    ) -> Tensor:
        """
        Forward pass.

        Parameters:
            inputs: input features
            init_state: initialized hidden state
            adj: adjacency matrix
            teacher_forcing_ratio: probability of using teacher forcing
        
        Return:
            outputs: output
        
        Shape:
            inputs: (B, T, N, C)
            init_state: (num_layers, B, N*D)
            outputs: (T, B, N*out_dim)
        """
        
        inputs = inputs[:, :, :, 0].unsqueeze(-1).permute(1, 0, 2, 3)     # (T, B, N, C)
        t_window = inputs.shape[0]
        batch_size = inputs.shape[1]
        inputs = inputs.reshape(t_window, batch_size, -1)                 # (T, B, N*C)

        outputs = torch.zeros(t_window, batch_size, self.n_series*self.output_dim).to(inputs.device)

        # go symbol for decoder
        go_symbol = torch.zeros(batch_size, self.n_series, 1).to(inputs.device)
        decoder_input = go_symbol
        
        # Decoder
        for t in range(t_window):
            next_state = []
            for i in range(self.num_layers):
                state = init_state[i]
                decoder_output, state = self.decoder[i](decoder_input, state, adj)
                decoder_input = decoder_output
                next_state.append(state)
            init_state = torch.stack(next_state, dim=0)
            outputs[t] = decoder_output     # (B, N*out_dim)
            # scheduled sampling
            if random.random() < teacher_forcing_ratio:
                decoder_input = inputs[t]

        return outputs

class _DCGRUCell(nn.Module):
    """
    Graph Convolution Gated Recurrent Unit Cell.

    Parameters:
        input_dim: dimension of input features
        hid_dim: hidden dimension of rnn
        max_diffusion_step: max diffusion step
        n_series: number of nodes
        out_dim: output dimension
        activation: activation for cell state
        use_gc_for_ru: whether to use graph convolution inside rnn
    """
    def __init__(
        self, 
        input_dim: int, 
        hid_dim: int,
        max_diffusion_step: int, 
        n_series: int,
        out_dim: int = None, 
        activation: Callable = torch.tanh, 
        use_gc_for_ru: bool = True
    ):
        super(_DCGRUCell, self).__init__()

        self.hid_dim = hid_dim
        self.max_diffusion_step = max_diffusion_step
        self.n_series = n_series
        self.out_dim = out_dim
        self.activation = activation
        self.use_gc_for_ru = use_gc_for_ru
        

        self.gate = _DiffusionConv(
            input_dim=input_dim,
            hid_dim=hid_dim,
            output_dim=hid_dim * 2,
            n_series=n_series,  
            max_diffusion_step=max_diffusion_step
        )
        
        self.candidate = _DiffusionConv(
            input_dim=input_dim,
            hid_dim=hid_dim,
            output_dim=hid_dim,
            n_series=n_series,
            max_diffusion_step=max_diffusion_step
        )
        
        if out_dim is not None:
            self.project = nn.Linear(self.hid_dim, self.out_dim)

    def _calculate_random_walk_matrix(
        self, 
        adj_mx: Tensor
    ) -> Tensor:
        device = adj_mx.device
        adj_mx = adj_mx + torch.eye(int(adj_mx.shape[0])).to(device)
        d = torch.sum(adj_mx, 1)
        d_inv = 1. / d
        d_inv = torch.where(torch.isinf(d_inv).to(device), torch.zeros(d_inv.shape).to(device), d_inv)
        d_mat_inv = torch.diag(d_inv)
        random_walk_mx = torch.mm(d_mat_inv, adj_mx)
        return random_walk_mx
    
    @property
    def output_size(self) -> int:
        """compute the output size."""
        if self.out_dim is not None:
            output_size = self.n_series * self.out_dim
        else:
            output_size = self.n_series * self.hid_dim

        return output_size

    def forward(
        self, 
        inputs: Tensor, 
        state: Tensor,
        adj: Tensor
    ) -> Tuple[Tensor, Tensor]:
        
        """
        Forward pass.

        Parameters:
            inputs: input features
            state: hidden state
            adj: adjacency matrix

        Return:
            output: output
            new_state: current state
        
        Shape:
            inputs: (B, N*C)
            state: (B, N*D)
        """
        adj = self._calculate_random_walk_matrix(adj).t()
        output_size = 2 * self.hid_dim
        state = state.to(inputs.device)

        if self.use_gc_for_ru:
            fn = self.gate
        else:
            fn = self._fc
        
        # GRU
        value = torch.sigmoid(fn(inputs, state, adj, output_size, bias_start=1.0))
        value = torch.reshape(value, (-1, self.n_series, output_size))
        r, u = torch.split(value, self.hid_dim, dim = -1)
        r = r.reshape(-1, self.n_series * self.hid_dim)
        u = u.reshape(-1, self.n_series * self.hid_dim)
        c = self.candidate(inputs, r * state, adj, self.hid_dim)  # (B, N*D)

        if self.activation is not None:
            c = self.activation(c)
        output = new_state = u * state + (1 - u) * c

        # output projection
        if self.out_dim is not None:
            batch_size = inputs.shape[0]
            output = new_state.reshape(-1, self.hid_dim)  # (B*N, D)
            output = self.project(output)
            output = output.reshape(batch_size, self.output_size)  # (B, N*out_dim)

        return output, new_state
    
    @staticmethod
    def _concat(
        x: Tensor, 
        x_: Tensor
    ) -> Tensor:
        """concat tensor."""
        x_ = torch.unsqueeze(x_, 0)
        return torch.cat([x, x_], dim=0)

    def _fc(
        self, 
        inputs: Tensor, 
        state: Tensor, 
        output_size: int, 
        bias_start: float = 0.0
    ) -> None:
        pass

    def init_hidden_state(
        self, 
        batch_size: int
    ) -> Tensor:
        '''
        Initialization of hidden state.
        '''
        return torch.zeros(batch_size, self.n_series * self.hid_dim)

class _DiffusionConv(nn.Module):
    """
    Diffusion graph convolution.

    Parameters:
        input_dim: dimension of input features
        hid_dim: hidden dimension of rnn
        output_dim: dimension of output features
        n_series: number of nodes
        max_diffusion_step: diffusion step
        bias_start: bias start
    """
    def __init__(
        self,
        input_dim: int,
        hid_dim: int,
        output_dim: int,
        n_series: int,
        max_diffusion_step: int, 
        bias_start: float = 0.0
    ):
        super(_DiffusionConv, self).__init__()

        input_size = input_dim + hid_dim
        self.num_matrices = max_diffusion_step + 1
        self.n_series = n_series
        self.max_diffusion_step = max_diffusion_step
        self.bias_start = bias_start
        
        self.weight = nn.Parameter(torch.FloatTensor(input_size * self.num_matrices, output_dim))
        self.biases = nn.Parameter(torch.FloatTensor(output_dim,))
        self._reset_parameters()
 

    def _reset_parameters(self) -> None:
        nn.init.xavier_normal_(self.weight)
        nn.init.constant_(self.biases, val=self.bias_start)

    @staticmethod
    def _concat(
        x: Tensor, 
        x_: Tensor
    ) -> Tensor:
        """concat tensor."""
        x_ = torch.unsqueeze(x_, 0)
        return torch.cat([x, x_], dim=0)

    def forward(
        self, 
        inputs: Tensor, 
        state: Tensor,
        adj: Tensor,
        output_size: int,
        bias_start: float = 0.0
    ) -> Tensor:
        """
        Forward pass.

        Parameters:
            inputs: input features
            state: hidden state
            adj: adjacency matrix
            output_size: output size
            bias_start: bias start
        
        Shape:
            inputs: (B, N*C)
            state: (B, N*D)
        """

        batch_size = inputs.shape[0]
        inputs = inputs.reshape(batch_size, self.n_series, -1)
        state = state.reshape(batch_size, self.n_series, -1)
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.shape[2]

        x = inputs_and_state    # (B, N, C')
        x0 = x.permute(1, 2, 0) # (N, C', B)
        x0 = x0.reshape(self.n_series, batch_size * input_size)    # (N, B*C')
        x = x0.unsqueeze(0)     # (1, N, B*C')

        if self.max_diffusion_step == 0:
            pass
        else:
            x1 = torch.mm(adj, x0)
            x = self._concat(x, x1)
            for _ in range(2, self.max_diffusion_step + 1):
                x2 = 2 * torch.mm(adj, x1) - x0
                x = self._concat(x, x2)
                x1, x0 = x2, x1

        x = x.reshape(self.num_matrices, self.n_series, input_size, batch_size)
        x = x.permute(3, 1, 2, 0)  # (B, N, C, num_matrices)
        x = x.reshape(batch_size * self.n_series, input_size * self.num_matrices)

        x = torch.matmul(x, self.weight)  # (B * N, output_size)
        x = torch.add(x, self.biases)
        
        return x.reshape(batch_size, self.n_series * output_size)