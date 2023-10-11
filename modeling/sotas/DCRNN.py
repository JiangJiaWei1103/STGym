"""
Baseline method, DCRNN [ICLR, 2018].
Author: ChunWei Shen

Reference:
* Paper: https://arxiv.org/abs/1707.01926
* Code:
    * https://github.com/liyaguang/DCRNN
    * https://github.com/xlwang233/pytorch-DCRNN
"""
import random
import numpy as np
from typing import List, Any, Dict, Callable, Tuple, Optional

import torch
import torch.nn as nn
from torch import Tensor

class DCRNN(nn.Module):
    """
    DCRNN.

    Parameters:
        n_series: number of nodes
        num_layers: number of DCRNN layers
        rnn_units: hidden dimension
        encoder_input_dim: input dimension of encoder
        decoder_input_dim: input dimension of decoder
        out_dim: output dimension
        n_adjs: number of transition matrices
        max_diffusion_step: maximum diffusion step
        cl_decay_steps: control the decay rate of cl threshold
        use_curriculum_learning: if True, model is trained with scheduled sampling
        out_len: output sequence length
    """
    def __init__(
        self, 
        st_params: Dict[str, Any],
        out_len: int
    ):
        super(DCRNN, self).__init__()

        # Network parameters
        self.st_params = st_params
        self.out_len = out_len
        # hyperparameters of Spatial/Temporal pattern extractor
        n_series = self.st_params['n_series']
        num_layers = self.st_params['num_layers']
        rnn_units = self.st_params['rnn_units']
        encoder_input_dim = self.st_params['encoder_input_dim']
        decoder_input_dim = self.st_params['decoder_input_dim']
        n_adjs = self.st_params["n_adjs"]
        max_diffusion_step = self.st_params['max_diffusion_step']
        out_dim = self.st_params['out_dim']
        # Curriculum learning strategy, scheduled sampling
        self.cl_decay_steps = self.st_params['cl_decay_steps']
        self.use_curriculum_learning = self.st_params['use_curriculum_learning']

        # Encoder
        self.encoder = _Encoder(
            input_dim=encoder_input_dim,
            hid_dim=rnn_units,
            num_layers =num_layers,
            n_adjs=n_adjs,
            max_diffusion_step=max_diffusion_step,
            n_series=n_series)
        
        # Decoder
        self.decoder = _Decoder(
            input_dim=decoder_input_dim,
            output_dim=out_dim,
            out_len=out_len,
            hid_dim=rnn_units,
            num_layers=num_layers,
            n_adjs=n_adjs, 
            max_diffusion_step=max_diffusion_step,
            n_series=n_series)

    def forward(
        self,
        input: Tensor,
        As: List[Tensor],
        ycl: Optional[Tensor] = None,
        iteration: Optional[int] = None,
        **kwargs: Any,
    ) -> Tuple[Tensor, None, None]:
        """
        Forward pass.

        Parameters:
            input: input features
            As: list of adjacency matrices
            ycl: ground truth
            iteration: number of batches already run

        Shape:
            input: (B, P, N, C)
            As: each A with shape (2, |E|), where |E| denotes the number edges
            ycl: (B, Q, N, C)
            output: (B, Q, N)
        """

        input = input.permute(1, 0, 2, 3)                        # (T, B, N, C)
        ycl = ycl[:, :, :, 0].permute(1, 0, 2).unsqueeze(-1)     # (Q, B, N, C)

        batch_size = input.shape[1]

        # Encoder
        init_state = self.encoder.init_hidden(batch_size)
        output_hidden, _ = self.encoder(input, init_state, As)  # (num_layers, B, N*D)

        if self.training and self.use_curriculum_learning:
            teacher_forcing_ratio = self._compute_sampling_threshold(iteration)
        else:
            teacher_forcing_ratio = 0

        # Decoder
        outputs = self.decoder(ycl, output_hidden, As, teacher_forcing_ratio=teacher_forcing_ratio)
        outputs = outputs.permute(1, 0, 2)     # (B, Q, N)

        return outputs, None, None
    
    def _compute_sampling_threshold(
        self, 
        iteration: int
    ) -> float:
        """Compute scheduled sampling threshold."""
        return (self.cl_decay_steps 
                / (self.cl_decay_steps + np.exp(iteration / self.cl_decay_steps)))

class _Encoder(nn.Module):
    """DCRNN Encoder."""
    def __init__(
        self,
        input_dim: int,
        hid_dim: int,
        num_layers: int,
        n_adjs: int, 
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
                    input_dim = input_dim,
                    hid_dim = hid_dim, 
                    n_adjs = n_adjs,
                    max_diffusion_step = max_diffusion_step,
                    n_series = n_series))

    def forward(
        self, 
        inputs: Tensor, 
        init_state: Tensor,
        As: List[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass.

        Parameters:
            inputs: input features
            init_state: initialized hidden state
            As: list of adjacency matrices

        Return:
            output_hidden: layer-wise last hidden state
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
                _, state = self.encoder[i](current_inputs[t, ...], state, As)    # (B, N*D)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=0)  # (T, B, N*D)

        return output_hidden, current_inputs

    def init_hidden(
        self, 
        batch_size: int
    ) -> Tensor:
        """Initialization of hidden state."""
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.encoder[i].init_hidden_state(batch_size))

        return torch.stack(init_states, dim=0)  # (num_layers, B, N*D)

class _Decoder(nn.Module):
    """DCRNN Decoder."""
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        out_len: int,
        hid_dim: int,
        num_layers: int,
        n_adjs: int,
        max_diffusion_step: int,
        n_series: int,
    ):
        super(_Decoder, self).__init__()

        self.hid_dim = hid_dim
        self.n_series = n_series
        self.output_dim = output_dim
        self.out_len = out_len
        self.num_layers = num_layers
        self.decoder = nn.ModuleList()

        for layer in range(num_layers):
            input_dim = input_dim if layer == 0 else hid_dim
            output_dim = self.output_dim if layer == (num_layers - 1) else None
            self.decoder.append(
                _DCGRUCell(
                    input_dim=input_dim,
                    hid_dim=hid_dim,
                    n_adjs=n_adjs,
                    max_diffusion_step=max_diffusion_step,
                    n_series=n_series,
                    out_dim = output_dim))        

    def forward(
        self, 
        inputs: Tensor, 
        init_state: Tensor,
        As: List[Tensor],
        teacher_forcing_ratio: float = 0.5
    ) -> Tensor:
        """
        Forward pass.

        Parameters:
            inputs: layer-wise last hidden state of encoder
            init_state: initialized hidden state
            As: list of adjacency matrices
            teacher_forcing_ratio: probability to feed the previous ground truth as input
        
        Return:
            outputs: output
        
        Shape:
            inputs: (T, B, N, C)
            init_state: (num_layers, B, N*D)
            outputs: (Q, B, N*out_dim)
        """
        
        t_window = inputs.shape[0]
        batch_size = inputs.shape[1]
        inputs = inputs.reshape(t_window, batch_size, -1)  # (T, B, N*C)

        outputs = torch.zeros(t_window, batch_size, self.n_series * self.output_dim).to(inputs.device)

        # go symbol for decoder
        go_symbol = torch.zeros(batch_size, self.n_series, 1).to(inputs.device)
        decoder_input = go_symbol
        
        # Decoder
        for t in range(self.out_len):
            next_state = []
            for i in range(self.num_layers):
                state = init_state[i]
                decoder_output, state = self.decoder[i](decoder_input, state, As)
                decoder_input = decoder_output
                next_state.append(state)
            init_state = torch.stack(next_state, dim = 0)
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
        n_adjs: number of transition matrices
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
        n_adjs: int, 
        max_diffusion_step: int, 
        n_series: int,
        out_dim: int = None, 
        activation: Callable = torch.tanh, 
        use_gc_for_ru: bool = True
    ):
        super(_DCGRUCell, self).__init__()

        self.hid_dim = hid_dim
        self.n_adjs = n_adjs
        self.max_diffusion_step = max_diffusion_step
        self.n_series = n_series
        self.out_dim = out_dim
        self.activation = activation
        self.use_gc_for_ru = use_gc_for_ru
        

        self.gate = _DiffusionConv(
            input_dim = input_dim,
            hid_dim = hid_dim,
            output_dim = hid_dim * 2,
            n_series = n_series,
            n_adjs = self.n_adjs,   
            max_diffusion_step = max_diffusion_step)
        
        self.candidate = _DiffusionConv(
            input_dim = input_dim,
            hid_dim = hid_dim,
            output_dim = hid_dim,
            n_series = n_series,
            n_adjs = self.n_adjs, 
            max_diffusion_step = max_diffusion_step)
        
        if out_dim is not None:
            self.project = nn.Linear(self.hid_dim, self.out_dim)

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
        As: List[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        
        """
        Forward pass.

        Parameters:
            inputs: input features
            state: hidden state
            As: list of adjacency matrices

        Return:
            output: output
            new_state: current state
        
        Shape:
            inputs: (B, N*C)
            state: (B, N*D)
        """
        output_size = 2 * self.hid_dim
        state = state.to(inputs.device)

        if self.use_gc_for_ru:
            fn = self.gate
        else:
            fn = self._fc
        
        # GRU
        value = torch.sigmoid(fn(inputs, state, output_size, As, bias_start = 1.0))
        value = torch.reshape(value, (-1, self.n_series, output_size))
        r, u = torch.split(value, self.hid_dim, dim = -1)
        r = r.reshape(-1, self.n_series * self.hid_dim)
        u = u.reshape(-1, self.n_series * self.hid_dim)
        c = self.candidate(inputs, r * state, self.hid_dim, As)  # (B, N*D)

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
        '''Initialization of hidden state.'''
        return torch.zeros(batch_size, self.n_series * self.hid_dim)

class _DiffusionConv(nn.Module):
    """
    Diffusion graph convolution.

    Parameters:
        input_dim: dimension of input features
        hid_dim: hidden dimension of rnn
        output_dim: dimension of output features
        n_series: number of nodes
        n_adjs: number of transition matrices
        max_diffusion_step: max diffusion step
        bias_start: bias start
    """
    def __init__(
        self,
        input_dim: int,
        hid_dim: int,
        output_dim: int,
        n_series: int,
        n_adjs: int,   
        max_diffusion_step: int, 
        bias_start: float = 0.0
    ):
        super(_DiffusionConv, self).__init__()

        input_size = input_dim + hid_dim
        self.num_matrices = n_adjs * max_diffusion_step + 1
        self.n_series = n_series
        self.max_diffusion_step = max_diffusion_step
        self.bias_start = bias_start
        
        self.weight = nn.Parameter(torch.FloatTensor(input_size * self.num_matrices, output_dim))
        self.biases = nn.Parameter(torch.FloatTensor(output_dim,))
        self._reset_parameters()
 

    def _reset_parameters(self) -> None:
        nn.init.xavier_normal_(self.weight)
        nn.init.constant_(self.biases, val=self.bias_start)

    def forward(
        self, 
        inputs: Tensor, 
        state: Tensor, 
        output_size: int,
        As: List[Tensor],
        bias_start: float = 0.0
    ) -> Tensor:
        """
        Forward pass.

        Parameters:
            inputs: input features
            state: hidden state
            output_size: output size
            As: list of adjacency matrices
            bias_start: bias start
        
        Shape:
            inputs: (B, N*C)
            state: (B, N*D)
        """

        batch_size = inputs.shape[0]
        inputs = inputs.reshape(batch_size, self.n_series, -1)
        state = state.reshape(batch_size, self.n_series, -1)
        inputs_and_state = torch.cat([inputs, state], dim = 2)
        input_size = inputs_and_state.shape[2]

        x = inputs_and_state    # (B, N, C')
        x0 = x.permute(1, 2, 0) # (N, C', B)
        x0 = x0.reshape(self.n_series, batch_size * input_size)    # (N, B*C')
        x = x0.unsqueeze(0)     # (1, N, B*C')

        if self.max_diffusion_step == 0:
            pass
        else:
            for A in As:
                x1 = torch.mm(A, x0)
                x = torch.cat([x, x1.unsqueeze(0)], dim=0)
                for _ in range(2, self.max_diffusion_step + 1):
                    x2 = 2 * torch.mm(A, x1) - x0
                    x = torch.cat([x, x2.unsqueeze(0)], dim=0)
                    x1, x0 = x2, x1

        x = x.reshape(self.num_matrices, self.n_series, input_size, batch_size)
        x = x.permute(3, 1, 2, 0)  # (B, N, C, num_matrices)
        x = x.reshape(batch_size * self.n_series, input_size * self.num_matrices)  

        x = torch.matmul(x, self.weight)  # (B * N, output_size)
        x = torch.add(x, self.biases)
        
        return x.reshape(batch_size, self.n_series * output_size)