"""
Baseline method, STSGCN [AAAI, 2020].
Author: ChunWei Shen

Reference:
* Paper: https://ojs.aaai.org/index.php/AAAI/article/view/5438
* Code:
    * https://github.com/Davidham3/STSGCN
    * https://github.com/j1o2h3n/STSGCN
"""
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from modeling.module.layers import STSGCL
from modeling.module.common_layers import Linear2d

class STSGCN(nn.Module):
    """STSGCN framework.

    Args:
        n_layers: number of STSGCN layers
        h_dim: hidden dimension
        gcn_depth: depth of graph convolution
        n_series: number of nodes
        act: activation function
        temporal_emb_dim: dimension of temporal embedding
        spatial_emb_dim: dimension of spatial embedding
        in_dim: input feature dimension
        in_len: input sequence length
        device: device
        out_len: output sequence length
    """

    def __init__(self, in_dim: int, in_len: int, device: str, out_len: int, st_params: Dict[str, Any]) -> None:
        self.name = self.__class__.__name__
        super(STSGCN, self).__init__()

        # Network parameters
        self.st_params = st_params
        # Spatio-temporal pattern extractor
        self.n_layers = st_params["n_layers"]
        self.h_dim  = st_params["h_dim"]
        gcn_depth = st_params["gcn_depth"]
        self.n_series = st_params["n_series"]
        act = st_params["act"]
        temporal_emb_dim = st_params["temporal_emb_dim"]
        spatial_emb_dim = st_params["spatial_emb_dim"]
        self.in_len = in_len
        self.out_len = out_len
        
        # Model blocks
        # Input linear layer
        self.in_lin = Linear2d(in_dim, self.h_dim)
        # Mask matrix
        self.mask = nn.Parameter(torch.rand(3 * self.n_series, 3 * self.n_series).to(device), requires_grad=True)
        # Spatial-Temporal Synchronous Graph Convolutional Layer
        self.stsgcl = nn.ModuleList()
        for layer in range(self.n_layers):
            self.stsgcl.append(
                STSGCL(
                    in_dim=self.h_dim,
                    h_dim=self.h_dim,
                    gcn_depth=gcn_depth,
                    n_series=self.n_series,
                    t_window=self.in_len,
                    act=act,
                    t_emb_dim=temporal_emb_dim,
                    s_emb_dim=spatial_emb_dim
                )
            )
            self.in_len -= 2
        # Output layers
        self.output_layers = nn.ModuleList()
        for _ in range(self.out_len):
            self.output_layers.append(
                nn.Sequential(
                    Linear2d(self.in_len * self.h_dim, 128),
                    nn.ReLU(),
                    Linear2d(128, 1),
                )
            )

    def forward(self, x: Tensor, As: List[Tensor], **kwargs: Any) -> Tuple[Tensor, None, None]:
        """Forward pass.

        Args:
            x: input sequence
            As: list of adjacency matrices

        Returns:
            output: prediction

        Shape:
            x: (B, P, N, C)
            As: each A with shape (N, N)
            output: (B, Q, N)
        """
        x = x.permute(0, 3, 2, 1)           # (B, C, N, P)
        x = torch.relu(self.in_lin(x))      # (B, h_dim, N, P)
        x = x.permute(0, 3, 2, 1)           # (B, P, N, h_dim)

        # Localized Spatial-Temporal Graph
        As = self._construct_adj(As[0]).to(x.device)
        A = self.mask * As

        # Spatial-Temporal Synchronous Graph Convolutional Layer
        for i in range(self.n_layers):
            x = self.stsgcl[i](x, A)        # (B, T - 2 * layer_num, N, h_dim)
        x = x.permute(0, 1, 3, 2)           # (B, T - 2 * layer_num, h_dim, N)
        x = x.reshape(-1, self.in_len * self.h_dim, self.n_series, 1)
        
        # Output layers
        x_out = []
        for i in range(self.out_len):
            output = self.output_layers[i](x)      # (B, 1, N)
            x_out.append(output.squeeze())
        output = torch.stack(x_out, dim=1)         # (B, Q, N)

        return output, None, None
    
    def _construct_adj(self, A: Tensor, steps: int = 3) -> Tensor:
        """Localized Spatial-Temporal Graph Construction.

        Args:
            A: adjacency matrix
            steps: number of step

        Shape:
            A: (N, N)
            adj: (step * N, step * N)
        """
        N = len(A)
        adj = torch.zeros([N * steps] * 2, dtype=torch.float32)

        for i in range(steps):
            adj[i * N:(i + 1) * N, i * N:(i + 1) * N] = A

        for i in range(N):
            for k in range(steps - 1):
                adj[k * N + i, (k + 1) * N + i] = 1
                adj[(k + 1) * N + i, k * N + i] = 1

        for i in range(len(adj)):
            adj[i, i] = 1

        return adj