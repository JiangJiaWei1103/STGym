"""
Self-adaptive graph structure learner.
Author: JiaWei Jiang, ChunWei Shen
"""
import numpy as np
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .spatial_layers import HyperGCN

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


class GTSGSLearner(nn.Module):
    """Graph structure learner of GTS."""

    def __init__(self, fc_in_dim: int, n_series: int, temperature: int) -> None:
        super(GTSGSLearner, self).__init__()
        
        # Network parameters
        self.n_series = n_series
        self.temperature = temperature
        kernel_size = 10
        conv_h_dim = 8
        conv_out_dim = 16
        emb_dim = 100

        # Model blocks
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=conv_h_dim, kernel_size=kernel_size),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Conv1d(in_channels=conv_h_dim, out_channels=conv_out_dim, kernel_size=kernel_size),
            nn.ReLU(),
            nn.BatchNorm1d(16),

        )
        self.fc = nn.Sequential(
            nn.Linear(fc_in_dim, emb_dim),
            nn.ReLU(),
            nn.BatchNorm1d(emb_dim)
        )
        self.output = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, 2)
        )
        # Generate off-diagonal interaction graph
        off_diag = np.ones([n_series, n_series])
        rel_rec = np.array(self._encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        rel_send = np.array(self._encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
        self.rel_rec = torch.FloatTensor(rel_rec)
        self.rel_send = torch.FloatTensor(rel_send)

    def forward(self, node_features: Tensor) -> Tensor:
        """Forward pass.

        Returns:
            As: self-adaptive adjacency matrix

        Shape:
            As: (N, N)
        """
        x = node_features.transpose(1, 0).view(self.n_series, 1, -1)
        x = self.conv(x)
        x = x.view(self.n_series, -1)
        x = self.fc(x)

        receivers = torch.matmul(self.rel_rec.to(x.device), x)
        senders = torch.matmul(self.rel_send.to(x.device), x)
        x = torch.cat([senders, receivers], dim=1)
        x = self.output(x)

        As = self._gumbel_softmax(x, temperature=self.temperature, hard=True)
        As = As[:, 0].clone().reshape(self.n_series, -1)
        mask = torch.eye(self.n_series, self.n_series).bool().to(x.device)
        As.masked_fill_(mask, 0)

        As = self._calculate_random_walk_matrix(As).t()

        return As
    
    def _encode_onehot(self, labels: np.array) -> np.array:
        """Encode onehot."""
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)

        return labels_onehot
    
    def _sample_gumbel(self, logits: Tensor, eps: float = 1e-20) -> Tensor:
        """Sample gumbel."""
        U = torch.rand(logits.size()).to(logits.device)
        return -torch.autograd.Variable(torch.log(-torch.log(U + eps) + eps))

    def _gumbel_softmax_sample(self, logits: Tensor, temperature: int, eps: float = 1e-10) -> Tensor:
        """Gumbel softmax sample."""
        sample = self._sample_gumbel(logits, eps=eps)
        y = logits + sample

        return F.softmax(y / temperature, dim=-1)

    def _gumbel_softmax(self, logits: Tensor, temperature: int, hard: bool = False, eps: float = 1e-10) -> Tensor:
        """Sample from the Gumbel-Softmax distribution and optionally discretize.

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
            y_hard = torch.zeros(*shape).to(logits.device)
            y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
            y = torch.autograd.Variable(y_hard - y_soft.data) + y_soft
        else:
            y = y_soft

        return y
    
    def _calculate_random_walk_matrix(self, adj: Tensor) -> Tensor:
        """Calculate random walk matrix."""
        adj = adj + torch.eye(int(adj.shape[0])).to(adj.device)
        d = torch.sum(adj, 1)
        d_inv = 1. / d
        d_inv = torch.where(torch.isinf(d_inv).to(adj.device), torch.zeros(d_inv.shape).to(adj.device), d_inv)
        d_mat_inv = torch.diag(d_inv)
        adj = torch.mm(d_mat_inv, adj)

        return adj


class DGCRNGSLearner(nn.Module):
    """Graph structure learner of DGCRN."""

    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        mid_dim: int,
        depth: int,
        n_series: int,
        node_emb_dim: int,
        act_alpha: float,
        alpha: float,
        gamma: float
    ) -> None:
        super(DGCRNGSLearner, self).__init__()

        # Network parameters
        self.act_alpha = act_alpha

        # Model blocks
        self.src_emb = nn.Embedding(n_series, node_emb_dim)
        self.tgt_emb = nn.Embedding(n_series, node_emb_dim)
        # Encoder
        self.en_src_gcn = HyperGCN(in_dim, h_dim, mid_dim, node_emb_dim, depth, alpha, gamma)
        self.en_src_gcn_t = HyperGCN(in_dim, h_dim, mid_dim, node_emb_dim, depth, alpha, gamma)
        self.en_tgt_gcn = HyperGCN(in_dim, h_dim, mid_dim, node_emb_dim, depth, alpha, gamma)
        self.en_tgt_gcn_t = HyperGCN(in_dim, h_dim, mid_dim, node_emb_dim, depth, alpha, gamma)
        # Decoder
        self.de_src_gcn = HyperGCN(in_dim, h_dim, mid_dim, node_emb_dim, depth, alpha, gamma)
        self.de_src_gcn_t = HyperGCN(in_dim, h_dim, mid_dim, node_emb_dim, depth, alpha, gamma)
        self.de_tgt_gcn = HyperGCN(in_dim, h_dim, mid_dim, node_emb_dim, depth, alpha, gamma)
        self.de_tgt_gcn_t = HyperGCN(in_dim, h_dim, mid_dim, node_emb_dim, depth, alpha, gamma)
    
    def forward(self, node_idx: Tensor, x_hyper: Tensor, As: List[Tensor], gsl_type: str) -> Tensor:
        """Forward pass.

        Args:
            node_idx: node indices
            x_hyper: node feature matrix
            As: list of predefined adjacency matrix

        Returns:
            A: dynamic adjacency matrix

        Shape:
            node_idx: (N, )
            A: (N, N)
        """
        e1 = self.src_emb(node_idx)
        e2 = self.tgt_emb(node_idx)

        if gsl_type == "encoder":
            filter1 = self.en_src_gcn(x_hyper, As[0]) + self.en_src_gcn_t(x_hyper, As[1])
            filter2 = self.en_tgt_gcn(x_hyper, As[0]) + self.en_tgt_gcn_t(x_hyper, As[1])
        else:
            filter1 = self.de_src_gcn(x_hyper, As[0]) + self.de_src_gcn_t(x_hyper, As[1])
            filter2 = self.de_tgt_gcn(x_hyper, As[0]) + self.de_tgt_gcn_t(x_hyper, As[1])

        m1 = torch.tanh(self.act_alpha * torch.mul(e1, filter1))
        m2 = torch.tanh(self.act_alpha * torch.mul(e2, filter2))

        A_soft = F.relu(F.tanh(self.act_alpha * (m1 @ m2.transpose(1, 2) - m2 @ m1.transpose(1, 2))))

        A, AT = self._normalize(A_soft), self._normalize(A_soft.transpose(1, 2))

        return A, AT

    def _normalize(self, A: Tensor) -> Tensor:
        """Normalize adjacency matrix."""

        A = A + torch.eye(A.size(-1)).to(A.device)
        A = A / torch.unsqueeze(A.sum(-1), -1)

        return A


class MegaCRNGSLearner(nn.Module):
    """Graph structure learner of MegaCRN."""

    def __init__(self) -> None:
        super(MegaCRNGSLearner, self).__init__()

    def forward(self, memory: Tensor, w1: Tensor, w2: Tensor) -> Tensor:
        """Forward pass.

        Returns:
            As: list of adjacency matrix

        Shape:
            As: each A with shape (N, N)
        """
        src_emb = torch.matmul(w1, memory)
        tgt_emb = torch.matmul(w2, memory)
        A1 = F.softmax(F.relu(torch.mm(src_emb, tgt_emb.T)), dim=-1)
        A2 = F.softmax(F.relu(torch.mm(tgt_emb, src_emb.T)), dim=-1)
        As = [A1, A2]

        return As