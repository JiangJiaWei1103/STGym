"""
Baseline method, DCRNN [ICLR, 2018].
Author: JiaWei Jiang

Reference:
* Paper: https://arxiv.org/abs/1707.01926
* Code:
    * https://github.com/liyaguang/DCRNN
    * https://github.com/xlwang233/pytorch-DCRNN
"""
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from modeling.sotas.layers import DCGRU


class DCRNN(nn.Module):
    """DCRNN framework.

    Args:
        n_layers: number of DCRNN layers
        enc_in_dim: input dimension of encoder
        dec_in_dim: input dimension of decoder
        h_dim: hidden dimension
        out_dim: output dimension
        n_adjs: number of transition matrices
        max_diffusion_step: maximum diffusion step
        use_curriculum_learning: if True, model is trained with
            scheduled sampling
        cl_decay_steps: control the decay rate of cl threshold
        out_len: output sequence length
    """

    def __init__(self, st_params: Dict[str, Any], out_len: int) -> None:
        self.name = self.__class__.__name__
        super(DCRNN, self).__init__()

        # Network parameters
        # Spatio-temporal pattern extractor
        self.st_params = st_params
        n_layers = self.st_params["n_layers"]
        enc_in_dim = self.st_params["enc_in_dim"]
        dec_in_dim = self.st_params["dec_in_dim"]
        h_dim = self.st_params["h_dim"]
        out_dim = self.st_params["out_dim"]
        n_adjs = self.st_params["n_adjs"]
        max_diffusion_step = self.st_params["max_diffusion_step"]
        # Curriculum learning strategy, scheduled sampling
        self.use_curriculum_learning = self.st_params["use_curriculum_learning"]
        self.cl_decay_steps = self.st_params["cl_decay_steps"]

        # Model blocks
        # Encoder
        self.encoder = _Encoder(
            in_dim=enc_in_dim,
            h_dim=h_dim,
            n_layers=n_layers,
            n_adjs=n_adjs,
            max_diffusion_step=max_diffusion_step,
            act=None,
        )
        # Decoder
        self.decoder = _Decoder(
            in_dim=dec_in_dim,
            h_dim=h_dim,
            out_dim=out_dim,
            out_len=out_len,
            n_layers=n_layers,
            n_adjs=n_adjs,
            max_diffusion_step=max_diffusion_step,
            act=None,
        )

    def forward(
        self, x: Tensor, As: List[Tensor], ycl: Optional[Tensor] = None, iteration: Optional[int] = None
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
        # Encoder
        hs = self.encoder(x, As)  # (n_layers, B, N, h_dim)

        # Decoder
        if self.training and self.use_curriculum_learning:
            teacher_forcing_ratio = self._compute_sampling_threshold(iteration)
        else:
            teacher_forcing_ratio = 0
        output = self.decoder(hs, As, ycl, teacher_forcing_ratio)  # (B, Q, N)

        return output, None, None

    def _compute_sampling_threshold(self, iteration: int) -> float:
        """Compute scheduled sampling threshold."""
        thres = self.cl_decay_steps / (self.cl_decay_steps + np.exp(iteration / self.cl_decay_steps))

        return thres


class _Encoder(nn.Module):
    """DCRNN encoder."""

    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        n_layers: int,
        n_adjs: int = 2,
        max_diffusion_step: int = 2,
        act: Optional[str] = None,
    ):
        super(_Encoder, self).__init__()

        # Model blocks
        self.encoder = nn.ModuleList()
        for layer in range(n_layers):
            in_dim = in_dim if layer == 0 else h_dim
            self.encoder.append(
                DCGRU(in_dim=in_dim, h_dim=h_dim, n_adjs=n_adjs, max_diffusion_step=max_diffusion_step, act=act)
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
            As: each A with shape (2, |E|), where |E| denotes the
                number edges
            hs: (n_layers, B, N, h_dim)
        """
        hs = []
        for encoder_layer in self.encoder:
            x, h_last = encoder_layer(x, As, h_0=None)  # (B, N, h_dim)
            hs.append(h_last)
        hs = torch.stack(hs)  # (n_layers, B, N, h_dim)

        return hs


class _Decoder(nn.Module):
    """DCRNN decoder."""

    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        out_dim: int,
        out_len: int,
        n_layers: int,
        n_adjs: int = 2,
        max_diffusion_step: int = 2,
        act: Optional[str] = None,
    ):
        super(_Decoder, self).__init__()

        # Network parameters
        self.out_len = out_len

        # Model blocks
        self.decoder = nn.ModuleList()
        for layer in range(n_layers):
            in_dim = in_dim if layer == 0 else h_dim
            self.decoder.append(
                DCGRU(in_dim=in_dim, h_dim=h_dim, n_adjs=n_adjs, max_diffusion_step=max_diffusion_step, act=act)
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
