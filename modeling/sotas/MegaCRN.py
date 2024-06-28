"""
Baseline method, MegaCRN [AAAI, 2023].
Author: ChunWei Shen

Reference:
* Paper: https://arxiv.org/abs/2212.05989
* Code: https://github.com/deepkashiwa20/MegaCRN
"""
import random
import numpy as np
from typing import List, Any, Dict, Tuple, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from torch.nn.modules.loss import _Loss
from utils.scaler import MaxScaler, StandardScaler

from modeling.module.layers import DCGRU
from modeling.module.common_layers import Memory
from modeling.module.gs_learner import MegaCRNGSLearner

class MegaCRN(nn.Module):
    """MegaCRN framework.

        Args:
            out_dim: output dimension
            out_len: output sequence length
            n_layers: number of MegaCRN layers
            n_series: number of series
            h_dim: hidden dimension
            enc_in_dim: input dimension of encoder
            dec_in_dim: input dimension of decoder
            n_adjs: number of transition matrices 
            max_diffusion_step: maximum diffusion step
            use_curriculum_learning: if True, model is trained with
                scheduled sampling
            cl_decay_steps: control the decay rate of cl threshold
            mem_num: number of memory items
            mem_dim: dimension of each memory item
    """
    
    def __init__(
        self,
        out_dim: int,
        out_len: int,
        st_params: Dict[str, Any],
        mem_params: Dict[str, Any],
        loss_params: Dict[str, Any],
    ) -> None:
        self.name = self.__class__.__name__
        super(MegaCRN, self).__init__()

        # Network parameters
        self.st_params = st_params
        self.mem_params = mem_params
        self.loss_params = loss_params
        self.out_len = out_len
        # Spatio-temporal pattern extractor
        self.n_layers = st_params["n_layers"]
        n_series = st_params["n_series"]
        h_dim = st_params["h_dim"]
        enc_in_dim = st_params["enc_in_dim"]
        dec_in_dim = st_params["dec_in_dim"]
        n_adjs = st_params["n_adjs"]
        max_diffusion_step = st_params["max_diffusion_step"]
        # Curriculum learning strategy, scheduled sampling
        self.use_curriculum_learning = self.st_params["use_curriculum_learning"]
        self.cl_decay_steps = self.st_params["cl_decay_steps"]
        # Memory
        mem_num = self.mem_params["mem_num"]
        mem_dim = self.mem_params["mem_dim"]
        # Custom loss
        self.lamb = self.loss_params['lamb']
        self.lamb1 = self.loss_params['lamb1']

        # Model blocks
        # Memory
        self.memory = Memory(mem_num=mem_num, mem_dim=mem_dim, h_dim=h_dim, n_series=n_series)
        # Meta-Graph Learner
        self.gs_learner = MegaCRNGSLearner()
        # Encoder
        self.encoder = _Encoder(
            in_dim=enc_in_dim,
            h_dim=h_dim,
            n_layers=self.n_layers,
            n_adjs=n_adjs,
            max_diffusion_step=max_diffusion_step,
        )
        # Decoder
        self.decoder = _Decoder(
            in_dim=dec_in_dim,
            h_dim=h_dim + mem_dim,
            out_dim=out_dim,
            out_len=out_len,
            n_layers=self.n_layers,
            n_adjs=n_adjs,
            max_diffusion_step=max_diffusion_step,
        )
            
    def forward(
        self,
        x: Tensor,
        As: Optional[List[Tensor]] = None,
        ycl: Optional[Tensor] = None,
        iteration: Optional[int] = None,
        **kwargs: Any,
    ) -> Tuple[Tensor, Tensor, Tuple[Tensor, Tensor, Tensor]]:
        """Forward pass.

        Args:
            x: input sequence
            As: list of adjacency matrices
            ycl: ground truth observation
            iteration: current iteration number

        Shape:
            x: (B, P, N, C)
            ycl: (B, Q, N, C)
            output: (B, Q, N)
        """
        # Spatio-Temporal Meta-Graph Learner
        Memory, We1, We2 = self.memory()
        As = self.gs_learner(Memory, We1, We2)

        # Encoder
        hs = self.encoder(x[..., :1], As)    # (B, T, N, hid_dim)
        hs = hs[:, -1, :, :]        # (B, N, hid_dim)   

        # Meta-node vector
        h_att, query, pos, neg = self.memory(hs)
        hs = torch.cat([hs, h_att], dim=-1)

        # Decoder
        if self.training and self.use_curriculum_learning:
            teacher_forcing_ratio = self._compute_sampling_threshold(iteration)
        else:
            teacher_forcing_ratio = 0
        hs = torch.stack([hs] * self.n_layers, dim=0)
        output = self.decoder(hs, As, ycl, teacher_forcing_ratio)  # (B, Q, N)
        
        return output, h_att, (query, pos, neg)
    
    def _compute_sampling_threshold(self, iteration: int) -> float:
        """Compute scheduled sampling threshold."""
        thres = self.cl_decay_steps / (self.cl_decay_steps + np.exp(iteration / self.cl_decay_steps))

        return thres
    
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
        query, pos, neg = aux_output[1]

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
        aux_output: Tensor,
        scaler: Union[MaxScaler, StandardScaler],
        criterion: _Loss,
    ) -> Tuple[float, Tensor, Tensor]:
        """Custom loss function.

        Args:
            y_pred: model prediction
            y_true: ground truth
            aux_output: auxiliary output
            scaler: scaler
        """
        y_pred_inv = scaler.inverse_transform(y_pred)
        y_true_inv = scaler.inverse_transform(y_true)
        
        query, pos, neg = aux_output[1]

        separate_loss = nn.TripletMarginLoss(margin=1.0)
        compact_loss = nn.MSELoss()

        loss1 = criterion(y_pred_inv, y_true_inv)
        loss2 = separate_loss(query, pos.detach(), neg.detach())
        loss3 = compact_loss(query, pos.detach())

        loss = loss1 + self.lamb * loss2 + self.lamb1 * loss3

        return loss, y_pred, y_true
        
    
class _Encoder(nn.Module):
    """MegaCRN encoder."""

    def __init__(self, in_dim: int, h_dim: int, n_layers: int, n_adjs: int = 2, max_diffusion_step: int = 2) -> None:
        super(_Encoder, self).__init__()

        # Model blocks
        self.encoder = nn.ModuleList()
        for layer in range(n_layers):
            in_dim = in_dim if layer == 0 else h_dim
            self.encoder.append(
                DCGRU(in_dim=in_dim, h_dim=h_dim, n_adjs=n_adjs, max_diffusion_step=max_diffusion_step)
            )

    def forward(self, x: Tensor, As: List[Tensor]) -> Tensor:
        """Forward pass.

        Args:
            x: input seqeunce
            As: list of adjacency matrices

        Returns:
            hs: layer-wise last hidden state

        Shape:
            x: (B, P, N, C)
            hs: (B, T, N, h_dim)
        """
        hs = x
        for encoder_layer in self.encoder:
            hs, _ = encoder_layer(hs, As, h_0=None)  # (B, T, N, h_dim)

        return hs


class _Decoder(nn.Module):
    """MegaCRN decoder."""

    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        out_dim: int,
        out_len: int,
        n_layers: int,
        n_adjs: int = 2,
        max_diffusion_step: int = 2,
    ):
        super(_Decoder, self).__init__()

        # Network parameters
        self.out_len = out_len

        # Model blocks
        self.decoder = nn.ModuleList()
        for layer in range(n_layers):
            in_dim = in_dim if layer == 0 else h_dim
            self.decoder.append(
                DCGRU(in_dim=in_dim, h_dim=h_dim, n_adjs=n_adjs, max_diffusion_step=max_diffusion_step)
            )
        self.out_proj = nn.Linear(h_dim, out_dim)

    def forward(self, hs: Tensor, As: List[Tensor], ycl: Tensor, teacher_forcing_ratio: float = 0.5) -> Tensor:
        """Forward pass.

        Args:
            hs: layer-wise last hidden state of encoder
            As: list of adjacency matrices
            ycl: groud truth observation
            teacher_forcing_ratio: probability to feed the previous
                ground truth as input

        Returns:
            output: prediction

        Shape:
            hs: (n_layers, B, N, h_dim)
            As: each A with shape (2, |E|), where |E| denotes the
                number edges
            ycl: (B, Q, N, h_dim)
            output: (B, Q, N)
        """
        _, batch_size, n_series = hs.shape[:-1]

        x = torch.zeros(batch_size, n_series, 1, device=hs.device)  # Go symbol
        output = []
        for q in range(self.out_len):
            hs_q = []
            for layer, decoder_layer in enumerate(self.decoder):
                x = x.unsqueeze(dim=1)  # Add time dim for compatibility
                x = torch.cat([x, ycl[:, q:q+1, :, 1:]], dim=-1)
                _, x = decoder_layer(x, As, h_0=hs[layer])  # (B, N, h_dim)
                hs_q.append(x)
            hs = torch.stack(hs_q)  # (n_layers, B, N, h_dim)

            output_q = self.out_proj(x)
            if random.random() < teacher_forcing_ratio:
                # Use ground truth as input
                x = ycl[:, q, :, 0].unsqueeze(dim=-1)  # (B, N, 1)
            else:
                x = output_q
            output.append(output_q)
        output = torch.cat(output, dim=-1).transpose(1, 2)  # (B, Q, N)

        return output