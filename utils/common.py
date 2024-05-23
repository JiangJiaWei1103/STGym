"""
Common utility functions used in training and evaluation processes.
Author: JiaWei Jiang
"""
import time
from collections import defaultdict
from decimal import Decimal
from typing import Dict, List

import numpy as np
import scipy as sp
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from scipy.sparse import coo_matrix, csr_matrix, linalg
from torch import Tensor
from torch.nn import Module


def count_params(model: Module) -> str:
    """Count number of parameters in model.

    Args:
        model: model instance

    Returns:
        The number of parameters in model, represented in scientific
        notation.
    """
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_params = f"{Decimal(str(n_params)):.4E}"

    return n_params


def dictconfig2dict(cfg: DictConfig) -> Dict:
    """Convert OmegaConf config object to primitive container."""
    cfg_resolved = OmegaConf.to_container(cfg, resolve=True)

    return cfg_resolved


class Profiler(object):
    """Profiler for probing time cost of the designated process."""

    t_start: float

    def __init__(self) -> None:
        self.proc_type = ""
        self.t_elapsed: Dict[str, List[float]] = defaultdict(list)

    def start(self, proc_type: str = "train") -> None:
        self.proc_type = proc_type
        self.t_start = time.time()

    def stop(self, record: bool = True) -> None:
        if record:
            self.t_elapsed[self.proc_type].append(time.time() - self.t_start)

    def summarize(self, log_wnb: bool = True) -> None:
        print("\n=====Profile Summary=====")
        for proc_type, t_elapsed in self.t_elapsed.items():
            t_avg = np.mean(t_elapsed)
            t_std = np.std(t_elapsed)
            print(f"{proc_type.upper()} takes {t_avg:.2f} ± {t_std:.2f} (sec/epoch)")

            if log_wnb:
                wandb.log({f"{proc_type}_time": {"avg": t_avg, "t_std": t_std}})


class AvgMeter(object):
    """Meter computing and storing current and average values.

    Args:
        name: name of the value to track
    """

    _val: float
    _sum: float
    _cnt: int
    _avg: float

    def __init__(self, name: str) -> None:
        self.val_name = name

        self._reset()

    def update(self, val: float, n: int = 1) -> None:
        self._val = val
        self._sum += val * n
        self._cnt += n
        self._avg = self._sum / self._cnt

    @property
    def val_cur(self) -> float:
        """Return current value."""
        return self._val

    @property
    def avg_cur(self) -> float:
        """Return current average value."""
        return self._avg

    def _reset(self) -> None:
        self._val = 0
        self._avg = 0
        self._sum = 0
        self._cnt = 0


# The utilities below are from https://github.com/nnzhan/Graph-WaveNet
def sym_norm(adj_mat: np.ndarray) -> Tensor:
    """Symmetrically normalize adjacency matrix."""
    adj_mat = sp.coo_matrix(adj_mat)
    rowsum = np.array(adj_mat.sum(1))  # Derive the degree
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj_mat_normed = adj_mat.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()
    adj_mat_normed = torch.tensor(adj_mat_normed)

    return adj_mat_normed


def asym_norm(adj_mat: np.ndarray) -> Tensor:
    """Asymmetrically normalize adjacency matrix."""
    adj_mat = sp.coo_matrix(adj_mat)
    rowsum = np.array(adj_mat.sum(1)).flatten()  # Derive the degree
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.0
    d_mat = sp.diags(d_inv)
    adj_mat_normed = d_mat.dot(adj_mat).astype(np.float32).todense()
    adj_mat_normed = torch.tensor(adj_mat_normed)

    return adj_mat_normed


# The utilities below are from https://github.com/chnsh/DCRNN_PyTorch
def calculate_random_walk_matrix(adj_mx: np.ndarray) -> coo_matrix:
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.0
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()

    return random_walk_mx


def calculate_scaled_laplacian(adj_mx: np.ndarray, lambda_max: int = 2, undirected: bool = True) -> csr_matrix:
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which="LM")
        lambda_max = lambda_max[0]  # type: ignore
    L = sp.csr_matrix(L)
    M, _ = L.shape
    Identity = sp.identity(M, format="csr", dtype=L.dtype)
    L = (2 / lambda_max * L) - Identity

    return L.astype(np.float32)


def calculate_normalized_laplacian(adj: np.ndarray) -> coo_matrix:
    """Derive normalized Laplacian matrix.

    L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    D = diag(A 1)
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    return normalized_laplacian
