"""
Self-adaptive graph structure learner.
Author: JiaWei Jiang, ChunWei Shen
"""
import numpy as np
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GWNetGSLearner(nn.Module):
    """Graph structure learner of GWNet.

    Args:
        n_nodes: number of nodes (i.e., series)
    """

    def __init__(self, n_nodes: int) -> None:
        super(GWNetGSLearner, self).__init__()

        self.src_emb = nn.Parameter(torch.randn(n_nodes, 10))
        self.tgt_emb = nn.Parameter(torch.randn(10, n_nodes))

    def forward(self) -> Tensor:
        """Forward pass.

        Returns:
            A: self-adaptive adjacency matrix

        Shape:
            A: (N, N)
        """
        A = F.softmax(F.relu(torch.mm(self.src_emb, self.tgt_emb)), dim=1)

        return A


class MTGNNGSLearner(nn.Module):
    """Graph structure learner of MTGNN.

    Args:
        n_nodes: number of nodes (i.e., series)
        node_emb_dim: node embedding dimension
        alpha: control the saturation rate of the activation function
        k: topk nearest neighbors are retained in sparsification
        static_feat_dim: dimension of static node features
    """

    def __init__(
        self, n_nodes: int, node_emb_dim: int, alpha: float, k: int, static_feat_dim: Optional[int] = None
    ) -> None:
        super(MTGNNGSLearner, self).__init__()

        # Network parameters
        self.n_nodes = n_nodes
        self.alpha = alpha
        self.k = k

        # Model blocks
        if static_feat_dim is None:
            self.src_emb = nn.Embedding(n_nodes, node_emb_dim)
            self.tgt_emb = nn.Embedding(n_nodes, node_emb_dim)
            self.src_lin = nn.Linear(node_emb_dim, node_emb_dim)
            self.tgt_lin = nn.Linear(node_emb_dim, node_emb_dim)
        else:
            self.src_lin = nn.Linear(static_feat_dim, node_emb_dim)
            self.tgt_lin = nn.Linear(static_feat_dim, node_emb_dim)

    def forward(self, node_idx: Tensor, node_feat: Optional[Tensor] = None) -> None:
        """Forward pass.

        Args:
            node_idx: node indices
            node_feat: static node feature matrix

        Returns:
            A: self-adaptive adjacency matrix

        Shape:
            node_idx: (N, )
            node_feat: (N, D), where D denotes the static feature dimension
            A: (N, N)
        """
        if node_feat is None:
            e1 = self.src_emb(node_idx)
            e2 = self.tgt_emb(node_idx)
        else:
            e1 = e2 = node_feat

        m1 = F.tanh(self.alpha * self.src_lin(e1))
        m2 = F.tanh(self.alpha * self.tgt_lin(e2))
        A_soft = F.relu(F.tanh(self.alpha * (m1 @ m2.T - m2 @ m1.T)))

        # Perform KNN-like sparsification
        A_doped = A_soft + torch.rand_like(A_soft) * 0.01
        topk_val, topk_idx = torch.topk(A_doped, self.k)  # Along the last dim
        mask = torch.zeros(A_doped.size()).to(A_doped.device)
        mask.scatter_(1, topk_idx, topk_val.fill_(1))
        A = A_soft * mask

        return A
   

class AGCRNGSLearner(nn.Module):
    """Graph structure learner of AGCRN.

    Args:
        n_nodes: number of nodes (i.e., series)
        cheb_k: order of chebyshev polynomial expansion
    """

    def __init__(self, n_nodes: int, cheb_k: int) -> None:
        super(AGCRNGSLearner, self).__init__()

        # Network parameters
        self.n_nodes = n_nodes
        self.cheb_k = cheb_k

    def forward(self, node_emb: Tensor) -> Tensor:
        """Forward pass.

        Args:
            node_emb: node embedding matrix

        Returns:
            A: self-adaptive adjacency matrix

        Shape:
            A: (k, N, N)
        """
        A_soft = F.softmax(F.relu(torch.mm(node_emb, node_emb.transpose(0, 1))), dim=1)

        As = [torch.eye(self.n_nodes).to(A_soft.device), A_soft]
        for k in range(2, self.cheb_k):
            As.append(torch.matmul(2 * A_soft, As[-1]) - As[-2])

        A = torch.stack(As, dim=0)

        return A