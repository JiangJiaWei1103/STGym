"""
Baseline method, DGCRN [TKDD, 2021].
Author: ChunWei Shen

Reference:
* Paper: https://arxiv.org/abs/2104.14917
* Code: https://github.com/tsinghua-fib-lab/Traffic-Benchmark
"""
import random
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from modeling.module.layers import DGCRM
from modeling.module.gs_learner import DGCRNGSLearner   

class DGCRN(nn.Module):
    """
    DGCRN.

    Args:
        in_dim: input feature dimension
        out_dim: output dimension
        out_len: output sequence length
        n_series: number of series
        node_emb_dim: dimension of node embeddings
        gsl_h_dim: hidden dimension of graph structure learner
        gsl_mid_dim: middle dimension of graph structure learner
        act_alpha: control the saturation rate of the activation function
        h_dim: hidden dimension
        gcn_depth: depth of graph convolution
        alpha: retaining ratio for preserving locality
        beta: ratio for dynamic gcn
        gamma: ratio for static gcn
        use_curriculum_learning: if True, model is trained with
            scheduled sampling
        cl_decay_steps: control the decay rate of cl threshold
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        out_len: int,
        st_params: Dict[str, Any],
        gsl_params: Dict[str, Any],
    ):
        super(DGCRN, self).__init__()

        # Network parameters
        self.st_params = st_params
        self.gsl_params = gsl_params
        # Graph learning layer
        n_series = gsl_params["n_series"]
        node_emb_dim = gsl_params["node_emb_dim"]
        gsl_h_dim = gsl_params["gsl_h_dim"]
        gsl_mid_dim = gsl_params["gsl_mid_dim"]
        act_alpha = gsl_params["act_alpha"]
        # Spatio-temporal pattern extractor
        h_dim = st_params["h_dim"]
        gcn_depth = st_params["gcn_depth"]
        alpha = st_params["alpha"]
        beta = st_params["beta"]
        gamma = st_params["gamma"]
        # Curriculum learning strategy, scheduled sampling
        self.use_curriculum_learning = st_params["use_curriculum_learning"]
        self.cl_decay_steps = st_params["cl_decay_steps"]
        self.out_len = out_len

        # Model blocks
        # dynamic adjacency matrix 
        cat_dim = in_dim + h_dim
        self.gs_learner = DGCRNGSLearner(
            in_dim=cat_dim,
            h_dim=gsl_h_dim,
            mid_dim=gsl_mid_dim,
            depth=gcn_depth,
            n_series=n_series,
            node_emb_dim=node_emb_dim,
            act_alpha=act_alpha,
            alpha=alpha,
            gamma=gamma,
        )
        # Encoder
        self.encoder = _Encoder(
            in_dim=in_dim,
            h_dim=h_dim,
            gcn_depth=gcn_depth,
            n_series=n_series,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )
        # Decoder
        self.decoder = _Decoder(
            in_dim=in_dim,
            h_dim=h_dim,
            gcn_depth=gcn_depth,
            n_series=n_series,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            out_dim=out_dim,
        )
    
    def forward(
        self,
        x: Tensor,
        As: List[Tensor],
        ycl: Tensor,
        iteration: Optional[int] = None,
        task_level: int = None,
        **kwargs: Any,
    ) -> Tuple[Tensor, None, None]:
        """Forward pass.

        Args:
            x: input sequence
            As: list of adjacency matrices
            ycl: ground truth observation
            iteration: current iteration number

        Shape:
            x: (B, P, N, C)
            As: each A with shape (2, |E|), where |E| denotes the
                number edges
            ycl: (B, Q, N, C)
            output: (B, Q, N)
        """
        task_level = self.out_len if task_level == None else task_level

        # Encoder
        hs = self.encoder(x, As, self.gs_learner)  # (n_layers, B, N, h_dim)

        # Decoder
        if self.training and self.use_curriculum_learning:
            teacher_forcing_ratio = self._compute_sampling_threshold(iteration)
        else:
            teacher_forcing_ratio = 0
        output = self.decoder(hs, As, ycl, task_level, teacher_forcing_ratio, self.gs_learner)  # (B, Q, N)

        return output, None, None
    
    def _compute_sampling_threshold(self, iteration: int) -> float:
        """Compute scheduled sampling threshold."""
        thres = self.cl_decay_steps / (self.cl_decay_steps + np.exp(iteration / self.cl_decay_steps))

        return thres


class _Encoder(nn.Module):
    """DGCRN encoder."""
    
    def __init__(
        self, in_dim: int, h_dim: int, gcn_depth: int, n_series: int, alpha: float, beta: float, gamma: float
    ) -> None:
        super(_Encoder, self).__init__()

        # Model blocks
        self.encoder = DGCRM(
            in_dim=in_dim,
            h_dim=h_dim, 
            gcn_depth=gcn_depth,
            n_series=n_series,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            gsl_type="encoder",
        )

    def forward(self, x: Tensor, As: List[Tensor], gs_learner: DGCRNGSLearner) -> Tensor:
        """Forward pass.

        Args:
            x: input seqeunce
            As: list of adjacency matrices

        Returns:
            h_last: last hidden state

        Shape:
            x: (B, P, N, C)
        """
        _, h_last = self.encoder(x, As, gs_learner, h_0=None)  # (B, N, h_dim)
        
        return h_last


class _Decoder(nn.Module):
    """DGCRN decoder."""

    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        gcn_depth: int,
        n_series: int,
        alpha: float,
        beta: float,
        gamma: float,
        out_dim: int,
    ) -> None:
        super(_Decoder, self).__init__()

        # Model blocks
        self.decoder = DGCRM(
            in_dim=in_dim,
            h_dim=h_dim, 
            gcn_depth=gcn_depth,
            n_series=n_series,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            gsl_type="decoder",
        )
        self.out_proj = nn.Linear(h_dim, out_dim)

    def forward(
        self,
        hs: Tensor,
        As: List[Tensor],
        ycl: Tensor,
        task_level: int,
        teacher_forcing_ratio: float,
        gs_learner: DGCRNGSLearner,
    ) -> Tensor:
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
            hs: (B, N, h_dim)
            ycl: (B, Q, N, h_dim)
            output: (B, Q, N)
        """
        batch_size, n_series = hs.shape[:-1]

        x = torch.zeros(batch_size, n_series, 1, device=hs.device)  # Go symbol
        output = []
        for q in range(task_level):
            x = x.unsqueeze(dim=1)  # Add time dim for compatibility
            x_tid = ycl[:, q, :, 1:].unsqueeze(dim=1)
            _, hs = self.decoder(torch.cat([x, x_tid], dim=-1), As, gs_learner, h_0=hs)  # (B, N, h_dim)

            output_q = self.out_proj(hs)
            if random.random() < teacher_forcing_ratio:
                # Use ground truth as input
                x = ycl[:, q, :, 0].unsqueeze(dim=-1)  # (B, N, 1)
            else:
                x = output_q
            output.append(output_q)
        output = torch.cat(output, dim=-1).transpose(1, 2)  # (B, Q, N)

        return output