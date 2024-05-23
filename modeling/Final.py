"""
HARDPurG framework.
Author: JiaWei Jiang
"""
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from metadata import N_DAYS_IN_WEEK

from .common import GLU, MixProp

# Temporary in-file configuration
LATENT_DGSL = False  # Learn latent DGS or not
LATENT_DGSL_SMOOTH = False  # Smooth DGS with running mean or not


class HARDPurG(nn.Module):
    """Master network architecture, HARDPurG.

    Args:
        sgsl_params: hyperparameters of static graph structure learner
        gspp_params: hyperparameters of graph structure post processor
        st_params: hyperparameters of spatio-temporal pattern extractor
        skip_out_dim: output dimension of skip connection to the output
            module
        out_dim: output dimension
        priori_gs: always ignored, exist for compatibility
    """

    def __init__(
        self,
        sgsl_params: Dict[str, Any],
        sgspp_params: Dict[str, Any],
        # ===
        # For hyperparameter study
        dgspp_params: Dict[str, Any],
        # ===
        st_params: Dict[str, Any],
        skip_out_dim: int,
        out_dim: int,
        priori_gs: Optional[List[Tensor]] = None,
    ):
        self.name = self.__class__.__name__
        super(HARDPurG, self).__init__()

        # Network parameters
        self.sgsl_params = sgsl_params
        self.sgspp_params = sgspp_params
        self.st_params = st_params
        self.skip_out_dim = skip_out_dim
        self.out_dim = out_dim
        # Static graph structure learner
        n_series = self.sgsl_params["n_series"]
        assert n_series == self.st_params["n_series"], "Number of time series should be consistent."
        node_emb_dim = self.sgsl_params["node_emb_dim"]
        # Spatio-temporal pattern extractor
        n_layers = self.st_params["n_layers"]
        t_window = self.st_params["t_window"]
        tran_state_h_dim = self.st_params["tran_state_h_dim"]
        rnn_h_dim = self.st_params["rnn_h_dim"]
        rnn_n_layers = self.st_params["rnn_n_layers"]
        rnn_dropout = self.st_params["rnn_dropout"]
        gconv_type = self.st_params["gconv_type"]
        hop_aware_rectify_fn = self.st_params["hop_aware_rectify_fn"]
        gconv_depth = self.st_params["gconv_depth"]
        gconv_h_dim = self.st_params["gconv_h_dim"]
        tid_emb_dim = self.st_params["tid_emb_dim"]
        diw_emb_dim = self.st_params["diw_emb_dim"]
        n_tids = self.st_params["n_tids"]
        # Common
        common_dropout = self.st_params["common_dropout"]

        # Auxiliary information
        self.idx = torch.arange(n_series)  # Node identifiers

        # Model blocks
        # Temporal pattern initializer
        self.tpi = _TemporalPatternInitializer(
            tran_state_h_dim=tran_state_h_dim,
            common_dropout=common_dropout,
            rnn_h_dim=rnn_h_dim,
            rnn_n_layers=rnn_n_layers,
            rnn_dropout=rnn_dropout,
            t_window=t_window,
        )
        self.init_proj = nn.Linear(rnn_h_dim, skip_out_dim)
        # Auxiliary information encoder
        self.aux_info_enc = _AuxInfoEncoder(node_emb_dim * 2, tid_emb_dim, diw_emb_dim, n_tids)
        # Dynamic graph structure learner
        if not LATENT_DGSL:
            self.dgsl = _SketchedGSLearner(static=False, node_emb_dim=rnn_h_dim)
            self.dgspp = _GSPostProcessor(act="relu", k=dgspp_params["k"], symmetric=False, norm="asym")
            # self.dgspp = _GSPostProcessor(
            #     act="relu", k=10, symmetric=False, norm="asym"
            # )
        # Static graph structure learner
        self.sgsl = _SketchedGSLearner(**sgsl_params)
        self.sgspp = _GSPostProcessor(**sgspp_params)
        # HARDPurG layers
        self.hardpurg_layers = nn.ModuleList()
        self.skip_projs = nn.ModuleList()
        for layer in range(n_layers):
            self.hardpurg_layers.append(
                _HARDPurGLayer(
                    dy_wt_dim=rnn_h_dim,
                    node_emb_dim=rnn_h_dim,
                    gconv_type=gconv_type,
                    hop_aware_rectify_fn=hop_aware_rectify_fn,
                    dgconv_in_dim=rnn_h_dim,
                    sgconv_in_dim=gconv_h_dim,
                    gconv_h_dim=gconv_h_dim,
                    gconv_depth=gconv_depth,
                    n_series=n_series,
                    aux_info_dim=self.aux_info_enc.aux_info_dim,
                )
            )
            # Skip projector
            skip_in_dim = gconv_h_dim * 2  # Consider both static and dynamic patterns
            self.skip_projs.append(nn.Linear(skip_in_dim, skip_out_dim))
        self.dropout = nn.Dropout(p=common_dropout)
        # Last projector
        self.last_proj = nn.Linear(gconv_h_dim, skip_out_dim)
        # Output projector
        self.out_proj = nn.Sequential(
            nn.Linear(skip_out_dim, skip_out_dim // 2),
            nn.ReLU(),
            nn.Linear(skip_out_dim // 2, out_dim),
        )

    def forward(
        self,
        x: Tensor,
        tid: Optional[Tensor] = None,
        diw: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass.

        Args:
            x: node feature matrix
            tid: time in day identifier
            diw: day in week identifier

        Returns:
            output: prediction
            Ad: dynamic graph structure
            As: static graph structure

        Shape:
            x: (B, T, N), where B is the batch_size, T is the lookback
                time window and N is the number of time series
            tid: (B, )
            diw: (B, )
            output: (B, out_dim, N), out_dim=\tau
            Ad: (B, N, N)
            As: (1, N, N), B is always equal to 1, because the graph
                structure is static through time
        """
        batch_size, t_window, n_series, n_feats = x.shape
        if n_feats > 1:
            tid = x[:, -1, 0, 1].int()
            diw = x[:, -1, 0, 2].int()
        x = x[..., 0]

        # Temporal pattern initializer
        x_init = self.tpi(x)

        # Dynamic graph structure learner
        if not LATENT_DGSL:
            Ad_soft, _ = self.dgsl(x_init)
            Ad = self.dgspp(Ad_soft)
        else:
            Ad = None

        # Static graph structure learner
        if self.idx.device != x.device:
            self.idx = self.idx.to(x.device)
        As_soft, x_node = self.sgsl(x=self.idx)  # (1, N, N), (1, N, node_emb_dim*2)
        As = self.sgspp(As_soft)  # (1, N, N)

        # Auxiliary information encoder
        x_e_aux = self.aux_info_enc(x_node.expand(batch_size, -1, -1), tid, diw)

        # HARDPurG layers
        x_skip = self.init_proj(x_init)  # (B, N, d_skip)
        h_purged = x_init
        for layer, hardpurg_layer in enumerate(self.hardpurg_layers):
            if layer == 0:
                # No latent representation is available
                h_latent_d = None
                h_latent_s = None
                Ad_soft = None

            (h_cur_cat, (h_latent_d, h_latent_s), h_purged, Ad_soft_run,) = hardpurg_layer(  # h_l
                x_e_aux,
                h_purged,  # Init. with x_init
                As_soft,
                As,
                h_latent_d,
                h_latent_s,
                Ad,
                Ad_soft,
            )

            # Skip connection projector
            h_cur_cat = self.dropout(h_cur_cat)
            x_skip = x_skip + self.skip_projs[layer](h_cur_cat)  # (B, N, d_skip)

        # Last projector
        h_purged = self.dropout(h_purged)
        x_skip = x_skip + self.last_proj(h_purged)

        # Output projector
        output = self.out_proj(x_skip)  # (B, N, out_dim)
        if self.out_dim == 1:
            # Single-step forecasting scenario
            output = torch.squeeze(output, dim=-1)
        else:
            output = output.transpose(1, 2)  # (B, out_dim, N)

        return output, Ad, As


class _TemporalPatternInitializer(nn.Module):
    """Node-agnostic temporal pattern initializer.

    Args:
        tran_state_h_dim: hidden dimension of transient state (C)
        common_dropout: common dropout
        rnn_h_dim: hidden dimension of RNN (C')
        rnn_n_layers: number of RNN layers
        rnn_dropout: dropout in RNN
        t_window: number of lookback time steps
    """

    def __init__(
        self,
        tran_state_h_dim: int = 32,
        common_dropout: float = 0.2,
        rnn_h_dim: int = 32,
        rnn_n_layers: int = 1,
        rnn_dropout: float = 0.0,
        t_window: int = 168,
    ):
        self.name = self.__class__.__name__
        super(_TemporalPatternInitializer, self).__init__()

        # Network parameters
        self.tran_state_h_dim = tran_state_h_dim
        self.common_dropout = common_dropout
        self.rnn_h_dim = rnn_h_dim
        self.rnn_n_layers = rnn_n_layers
        self.rnn_dropout = rnn_dropout
        self.t_window = t_window

        # Model blocks
        self.tran_state_proj = nn.Sequential(
            nn.Linear(1, tran_state_h_dim),
            GLU(tran_state_h_dim, tran_state_h_dim, dropout=common_dropout),
        )
        self.rnn_init = nn.GRU(
            tran_state_h_dim,
            rnn_h_dim,
            rnn_n_layers,
            batch_first=True,
            dropout=rnn_dropout,
        )
        self.tconv = nn.Conv1d(t_window, 32, kernel_size=1)
        self.mha = nn.MultiheadAttention(32, num_heads=1, batch_first=True)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Shape:
            x: (B, T, N)
            x_init: (B, N, C')
        """
        batch_size, t_window, n_series = x.shape

        x = x.transpose(1, 2)  # (B, N, T)
        x = self.tran_state_proj(torch.unsqueeze(x, dim=-1))  # (B, N, T, C)
        x = x.contiguous().view(batch_size * n_series, t_window, -1)
        h_all, h_last = self.rnn_init(x)  # (B * N, T, C'), (1, B * N, C')

        q = h_last.transpose(0, 1)  # (B * N, 1, C')
        k = self.tconv(h_all)  # (B * N, 32, C')
        v = k
        x_init, *_ = self.mha(q, k, v, need_weights=False)  # (B * N, 1, C')
        x_init = x_init.contiguous().view(batch_size, n_series, -1)  # (B, N, C')

        return x_init


class _AuxInfoEncoder(nn.Module):
    """Auxiliary information encoder.

    Args:
        node_emb_dim: dimension of static node embedding
        tid_emb_dim: dimension of time in day embedding
        diw_emb_dim: dimension of day in week embedding
        n_tids: number of times in day
    """

    def __init__(
        self,
        node_emb_dim: int,
        tid_emb_dim: Optional[int],
        diw_emb_dim: Optional[int],
        n_tids: int,
    ):
        self.name = self.__class__.__name__
        super(_AuxInfoEncoder, self).__init__()

        # Network parameters
        self.node_emb_dim = node_emb_dim
        self.tid_emb_dim = tid_emb_dim
        self.diw_emb_dim = diw_emb_dim

        # Model blocks
        self.aux_info_dim = node_emb_dim
        self.tid_emb, self.diw_emb = None, None
        if tid_emb_dim is not None:
            self.aux_info_dim += tid_emb_dim
            tid_wts_init = nn.Parameter(torch.empty(n_tids, tid_emb_dim))
            nn.init.xavier_uniform_(tid_wts_init)
            self.tid_emb = nn.Embedding.from_pretrained(tid_wts_init, freeze=False)
        if diw_emb_dim is not None:
            self.aux_info_dim += diw_emb_dim
            diw_wts_init = nn.Parameter(torch.empty(N_DAYS_IN_WEEK, diw_emb_dim))
            nn.init.xavier_uniform_(diw_wts_init)
            self.diw_emb = nn.Embedding.from_pretrained(diw_wts_init, freeze=False)
        self.aux_enc = nn.Sequential(
            nn.Linear(self.aux_info_dim, self.aux_info_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.aux_info_dim, self.aux_info_dim),
        )

    def forward(self, x_node: Tensor, tid: Optional[Tensor] = None, diw: Optional[Tensor] = None) -> Tensor:
        """Forward pass.

        Shape:
            x_node: (B, N, node_emb_dim * 2), node_emb_dim = d_node
            tid: (B, )
            diw: (B, )
            x_e_aux: (B, N, node_emb_dim*2 + tid_emb_dim + diw_emb_dim),
                tid_emb_dim = d_tid, diw_emb_dim = d_diw
        """
        batch_size, n_series, _ = x_node.shape  # (B, N, node_emb_dim * 2)

        # Embed auxiliary information
        x_e_cat = x_node
        if self.tid_emb is not None:
            assert tid is not None, "Time in day isn't fed into the model."
            x_tid = self.tid_emb(tid).unsqueeze(dim=1).expand(-1, n_series, -1)
            x_e_cat = torch.cat([x_e_cat, x_tid], dim=-1)
        if self.diw_emb is not None:
            assert diw is not None, "Day in week isn't fed into the model."
            x_diw = self.diw_emb(diw).unsqueeze(dim=1).expand(-1, n_series, -1)
            x_e_cat = torch.cat([x_e_cat, x_diw], dim=-1)

        # Encode auxiliary information
        x_e_cat_resid = x_e_cat
        x_e_aux = self.aux_enc(x_e_cat) + x_e_cat_resid

        return x_e_aux


class _SketchedGSLearner(nn.Module):
    """Sketched graph structure learner.

    GS learner aims at learning sketched graph structure, either static
    or dynamically changing over time.

    Args:
        static: if True, the learned GS is static through time
        node_emb_dim: dimension of static or dynamic node embedding
        n_series: number of time series
        debug: always ignored, exists for compatibility
            *Note: It's used to determine whether learnt GS is returned
                for further analysis
    """

    def __init__(
        self,
        static: bool,
        node_emb_dim: int,
        n_series: Optional[int] = None,
        debug: bool = False,
    ):
        self.name = self.__class__.__name__
        super(_SketchedGSLearner, self).__init__()

        # Network params
        self.static = static
        self.node_emb_dim = node_emb_dim
        self.n_series = n_series

        # Model blocks
        # Static node embedding
        if static:
            # Static node embedding
            assert n_series is not None, "Please specify number of time series for SGSL."
            self.emb_in = nn.Embedding(n_series, node_emb_dim)  # E_in
            self.emb_out = nn.Embedding(n_series, node_emb_dim)  # E_out
        # Graph structure constructor
        self.lin_in = nn.Sequential(nn.Linear(node_emb_dim, node_emb_dim), nn.Tanh())
        self.lin_out = nn.Sequential(nn.Linear(node_emb_dim, node_emb_dim), nn.Tanh())
        wts = torch.empty((node_emb_dim, node_emb_dim), dtype=torch.float32)
        wts = nn.init.xavier_uniform_(wts)
        self.gsl = nn.Parameter(wts, requires_grad=True)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.lin_in[0].weight, gain=torch.nn.init.calculate_gain("tanh"))
        torch.nn.init.xavier_uniform_(self.lin_out[0].weight, gain=torch.nn.init.calculate_gain("tanh"))

    def forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward pass.

        Shape:
            x: (B, N, C') for DGSL and (N, ) for SGSL
            A_soft: (B, N, N), B is always equal to 1 for SGSL
            x_node: (1, N, node_emb_dim * 2)
        """
        if self.static:
            # Static node embedding
            x_in = self.emb_in(x)
            x_out = self.emb_out(x)
            x_node = torch.cat((x_in, x_out), dim=1)  # (N, node_emb_dim*2)
            x_node = torch.unsqueeze(x_node, dim=0)  # (1, N, node_emb_dim*2)
        else:
            batch_size, n_series, n_feats = x.shape

        # Graph structure constructor
        if self.static:
            # Static graph structure shared among samples (i.e., static through time)
            A_soft = self.lin_in(x_in) @ self.gsl.to(x.device) @ torch.t(self.lin_out(x_out))  # (N, N)
            A_soft = torch.unsqueeze(A_soft, dim=0)  # (1, N, N)
        else:
            operands = (
                self.lin_in(x),
                self.gsl.to(x.device),
                self.lin_out(x).transpose(1, 2),
            )
            A_soft = torch.einsum("buc,cd,bdv->buv", *operands)  # (B, N, N)

        if self.static:
            return A_soft, x_node
        else:
            return A_soft, None


class _GSPostProcessor(nn.Module):
    """Apply post processing to refine sketched adjacency matrix.

    Args:
        act: non-linear activation applied to adjust edge weights
            *Note: Please specify None to retain linear activation
        k: number of top closest neighbors
            *Note: Please specify -1 (shortcut of number of time series
                minus 1) to retain full adjacency matrix
        symmetric: whether to symmetrize adjacency matrix
        norm: normalization method, the choices are as follows:
            {'sym', 'asym', None}
    """

    act: Optional[nn.Module]

    def __init__(
        self,
        act: str = "relu",
        k: int = 20,
        symmetric: bool = False,
        norm: str = "sym",
    ):
        self.name = self.__class__.__name__
        super(_GSPostProcessor, self).__init__()

        if act == "relu":
            self.act = nn.ReLU()
        elif act == "elu":
            self.act = nn.ELU()
        else:
            self.act = None
        self.k = k
        self.symmetric = symmetric
        self.norm = norm

    def forward(self, A: Tensor) -> Tensor:
        # Activation
        if self.act is not None:
            A = self._activate(A)

        # Sparsification
        if self.k != -1:
            A = self._sparsify(A)

        # Symmetrization
        if self.symmetric:
            A = self._symmetrize(A)

        # Normalization
        if self.norm is not None:
            A = self._normalize(A)

        return A

    def _activate(self, A: Tensor) -> Tensor:
        """Apply non-linear transformation.

        Args:
            A: adjacency matrices

        Returns:
            A: activated adjacency matrices
        """
        A = self.act(A)

        return A

    def _sparsify(self, A: Tensor) -> Tensor:
        """Apply KNN-based sparsification.

        Args:
            A: adjacency matrices

        Returns:
            A: sparsified adjacency matrices
        """
        device = A.device
        assert self.k < A.size(1), (
            f"KNN-based sparsification with {self.k} neighbors " "is disable, because k is larger than (#nodes - 1)."
        )

        A_doped = A + torch.rand_like(A) * 1e-4
        _, topk_idx = torch.topk(A_doped, self.k)  # Along the last dim
        mask = torch.zeros(A.size()).to(device)
        src = torch.ones(A.size()).to(device)
        mask.scatter_(2, topk_idx, src)
        A = A * mask

        return A

    def _symmetrize(self, A: Tensor) -> Tensor:
        """Apply symmetrization.

        Args:
            A: adjacency matrices

        Returns:
            A: symmetrized adjacency matrices
        """
        A = (A + A.transpose(1, 2)) / 2

        return A

    def _normalize(self, A: Tensor) -> Tensor:
        """Apply normalization.

        See https://math.stackexchange.com/questions/3035968/,
        https://github.com/tkipf/gcn/issues/142

        Args:
            A: adjacency matrices

        Shape:
            A: (B, N, N), where B is the batch size and N is the number
                of time series

        Returns:
            A: normalized adjacency matrices
        """
        assert A.dim() == 3, "Shape of A doesn't match (B, N, N)."

        if self.norm == "sym":
            A = A + torch.eye(A.size(1)).to(A.device)  # (B, N, N)
            D_inv = torch.sum(A, dim=2).pow(-0.5)
            D_inv[torch.isinf(D_inv)] = 0
            D_inv = torch.diag_embed(D_inv)
            A = torch.matmul(torch.matmul(D_inv, A), D_inv)
        elif self.norm == "asym":
            A = A + torch.eye(A.size(1)).to(A.device)  # (B, N, N)
            D = torch.sum(A, dim=2, keepdim=True)
            A = A / D
        elif self.norm == "softmax":
            pass

        return A


class _HARDPurGLayer(nn.Module):
    """Hop-aware rectified dynamic purging GNN layer.

    Args:
        dy_wt_dim: dimension of weight matrix for latent DGSL
            *Note: Currently disabled
        node_emb_dim: dimension of dynamic node embedding
        gconv_type: type of graph convolution
        dgconv_in_dim: input dimension of dynamic graph convolution
        sgconv_in_dim: input dimension of static graph convolution
        gconv_h_dim: hidden dimension of graph convolution
        gconv_depth: depth of graph convolution
        n_series: number of time series
        aux_info_dim: dimension of integrated auxliliary information
        hop_aware_rectify_fn: hop-aware rectifying function
            *Note: It's considered only when
                `gconv_type`="hop_aware_rectify"
        bn: if True, BN is applied after purging operation
    """

    def __init__(
        self,
        dy_wt_dim: int,  # (Deprecated)
        node_emb_dim: int,
        gconv_type: str,
        dgconv_in_dim: int,
        sgconv_in_dim: int,
        gconv_h_dim: int,
        gconv_depth: int,
        n_series: int,
        aux_info_dim: int,
        hop_aware_rectify_fn: Optional[str] = None,
        bn: bool = False,
    ):
        self.name = self.__class__.__name__
        super(_HARDPurGLayer, self).__init__()

        # Network parameters
        self.dy_wt_dim = dy_wt_dim
        self.node_emb_dim = node_emb_dim
        self.gconv_type = gconv_type
        self.dgconv_in_dim = dgconv_in_dim
        self.sgconv_in_dim = sgconv_in_dim
        self.gconv_h_dim = gconv_h_dim
        self.gconv_depth = gconv_depth
        self.n_series = n_series
        self.aux_info_dim = aux_info_dim
        self.hop_aware_rectify_fn = hop_aware_rectify_fn
        self.bn = bn

        # Model blocks
        # Latent dynamic graph structure learner
        if LATENT_DGSL:
            self.dgsl = _SketchedGSLearner(static=False, node_emb_dim=dy_wt_dim)
            self.gspp = _GSPostProcessor(act="relu", k=10, symmetric=False, norm="asym")
        # Auxiliary information regulator
        self.aux_info_reg = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(aux_info_dim, dgconv_in_dim), dim=None),
            nn.Sigmoid(),
        )
        # Hierarchical information purging mechanism
        self.dgcn = self._build_gconv(
            gconv_type,
            dgconv_in_dim,
            gconv_h_dim,
            gconv_depth,
            alpha=0.05,
            hop_aware_rectify_fn=hop_aware_rectify_fn,
        )
        self.sgcn = self._build_gconv(
            gconv_type,
            sgconv_in_dim,
            gconv_h_dim,
            gconv_depth,
            alpha=0.05,
            hop_aware_rectify_fn=hop_aware_rectify_fn,
        )
        self.purger1 = nn.Sequential(  # p1
            nn.utils.weight_norm(nn.Linear(gconv_h_dim, dgconv_in_dim), dim=None),
            nn.ReLU(),
        )
        self.purger2 = nn.Sequential(  # p2
            nn.utils.weight_norm(nn.Linear(gconv_h_dim, gconv_h_dim), dim=None),
            nn.ReLU(),
        )
        self.final_purger = nn.Sequential(  # p3
            nn.utils.weight_norm(nn.Linear(gconv_h_dim, node_emb_dim), dim=None),
            nn.ReLU(),
        )
        # BatchNorm
        if bn:
            self.bn1 = nn.BatchNorm1d(gconv_h_dim, affine=False)
            self.bn2 = nn.BatchNorm1d(gconv_h_dim, affine=False)

    def forward(
        self,
        x_e_aux: Tensor,
        h_prev: Tensor,
        As_soft: Tensor,
        As: Tensor,
        h_latent_d: Optional[List[Tensor]] = None,
        h_latent_s: Optional[List[Tensor]] = None,
        Ad: Optional[Tensor] = None,  # (Deprecated)
        Ad_soft: Optional[Tensor] = None,  # (Deprecated)
    ) -> Any:
        """Forward pass.

        Args:
            x_e_aux: integrated auxiliary information
            h_prev: dynamic node embedding of the previous layer (l-1)
                *Note: This embedding contains time-dependent message
            As_soft: SGS before post-processor is applied
            As: SGS after post-processor is applied
            h_latent_d: hop-aware intermediate node embedding over Ad
                from layer (l-1)
            h_latent_s: hop-aware intermediate node embedding over As
                from layer (l-1)

        Returns:
            h_cur_cat: final node embedding skipped to output
            h_latent_d: hop-aware intermediate node embedding over Ad
                for the next `_HARDPurGLayer`
            h_latent_s: hop-aware intermediate node embedding over As
                for the next `_HARDPurGLayer`
            h_l: output node embedding fed to the next `_HARDPurGLayer`

        Shape:
            x_e_aux: (B, N, aux_info_dim)
            h_prev: (B, N, C')
            h_l: (B, N, C')
            h_cur_cat: (B, N, d_conv*2)
        """
        batch_size, n_series, n_features = h_prev.shape

        # Dynamic graph structure learner
        Ad_soft_run = None
        if LATENT_DGSL:
            Ad_soft_layer, _ = self.dgsl(h_prev)
            if Ad_soft is None:
                Ad_soft_run = Ad_soft_layer
            else:
                Ad_soft_run = 0.95 * Ad_soft + 0.05 * Ad_soft_layer
            Ad = self.gspp(Ad_soft_run)
        else:
            assert Ad is not None

        # Auxiliary information regulator
        z = h_prev * self.aux_info_reg(x_e_aux)  # (B, N, C')

        # Hierarchical information purging mechanism
        h_cur_d, h_latent_d = self.dgcn(z, Ad, hs=h_latent_d)
        h_cur_d_ = z - self.purger1(h_cur_d)
        if self.bn:
            h_cur_d_ = h_cur_d_.transpose(1, 2)
            h_cur_d_ = self.bn1(h_cur_d_)
            h_cur_d_ = h_cur_d_.transpose(1, 2)

        h_cur_s, h_latent_s = self.sgcn(h_cur_d_, As.expand(batch_size, -1, -1), hs=h_latent_s)
        h_cur_s_ = h_cur_d_ - self.purger2(h_cur_s)
        if self.bn:
            h_cur_s_ = h_cur_s_.transpose(1, 2)
            h_cur_s_ = self.bn2(h_cur_s_)
            h_cur_s_ = h_cur_s_.transpose(1, 2)

        h_l = h_prev - self.final_purger(h_cur_s_)
        h_cur_cat = torch.cat([h_cur_d, h_cur_s], dim=-1)

        return h_cur_cat, (h_latent_d, h_latent_s), h_l, Ad_soft_run

    def _build_gconv(
        self,
        gconv_type: str,
        in_dim: int,
        h_dim: int,
        depth: int,
        alpha: float,
        dropout: Optional[float] = None,
        hop_aware_rectify_fn: Optional[str] = None,
    ) -> nn.Module:
        """Build and return a single graph convolution block.

        Args:
            gconv_type: type of graph convolution
            in_dim: input dimension of graph convolution
            h_dim: hidden dimension of graph convolution
            depth: depth of graph convolution
            alpha: retaining ratio of the original state of node
                features
            dropout: dropout ratio
            hop_aware_rectify_fn: hop-aware rectifying function
                *Note: It's considered only when
                    `gconv_type`="hop_aware_rectify"

        Returns:
            gconv: graph convolution block
        """
        gconv: nn.Module = None

        if gconv_type == "mixprop":
            assert hop_aware_rectify_fn is None, "Hop-aware aggregation is disableed in vanilla MixProp."
            gconv = MixProp(in_dim, h_dim, depth, alpha=alpha, dropout=dropout)
        elif gconv_type == "hop_aware_rectify":
            assert hop_aware_rectify_fn is not None, "Hop-aware rectifying function should be set."
            gconv = _HopAwareRecGConv(
                in_dim,
                h_dim,
                depth,
                alpha=alpha,
                dropout=dropout,
                hop_aware_rectify_fn=hop_aware_rectify_fn,
            )

        return gconv


class _HopAwareRecGConv(nn.Module):
    """Hop-aware rectified graph convolution module.

    Args:
        c_in: input channel number
        c_out: output channel number
        gcn_depth: depth of graph convolution
        alpha: retaining ratio of the original state of node features
        dropout: dropout ratio
        hop_aware_rectify_fn: hop-aware rectifying function
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        gcn_depth: int,
        alpha: float = 0.05,
        dropout: Optional[float] = None,
        hop_aware_rectify_fn: str = "mean",
    ):
        self.name = self.__class__.__name__
        super(_HopAwareRecGConv, self).__init__()

        # Network parameters
        self.c_in = c_in
        self.c_out = c_out
        self.gcn_depth = gcn_depth
        self.alpha = alpha
        self.hop_aware_rectify_fn = hop_aware_rectify_fn

        # Model blocks
        # Hop-aware rectifiers
        self.hop_aware_rectifiers = nn.ModuleList()
        for hop in range(gcn_depth):
            if hop_aware_rectify_fn == "linear":
                self.hop_aware_rectifiers.append(nn.Linear(c_in * 2, c_in))
            elif hop_aware_rectify_fn == "gru":
                self.hop_aware_rectifiers.append(nn.GRU(c_in, c_in, batch_first=True, dropout=0))
            elif hop_aware_rectify_fn == "glu":
                self.hop_aware_rectifiers.append(GLU(c_in, c_in, dropout=0.3))
        # Dropout
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        # Information selection
        self.l_slc = nn.utils.weight_norm(nn.Linear((gcn_depth + 1) * c_in, c_out), dim=None)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.l_slc.weight)

    def forward(self, x: Tensor, A: Tensor, hs: Optional[List[Tensor]] = None) -> Tuple[Tensor, List[Tensor]]:
        """Forward pass.

        Args:
            x: node features
            A: adjacency matrix
            hs: hop-aware intermediate node embedding over A from layer
                (l-1)

        Returns:
            h: final node embedding
            h_latent: hop-aware intermediate node embedding over A for
                the next `_HARDPurGLayer`

        Shape:
            x: (B, N, C'), (B, N, d_conv) or (B, N, *)
            A: (B, N, N), B is always equal to 1 for SGS
            h: (B, N, d_conv)
        """
        assert x.dim() == 3, "Shape of node features is wrong."
        batch_size, n_series, n_feats = x.shape

        # Hop-aware rectified graph convolution
        h_latent = []
        h_mix = x
        h = x
        for hop in range(self.gcn_depth):
            # Message passing and aggregation
            h = self.alpha * x + (1 - self.alpha) * torch.einsum(
                "bwc,bvw->bvc", (h, A)  # (B, N, C'), (B, N, N)
            )  # (B, N, C')

            if self.dropout is not None:
                h = self.dropout(h)

            # Hop-aware rectifying
            if self.hop_aware_rectify_fn == "linear":
                if hs is not None:
                    h = self.hop_aware_rectifiers[hop](torch.cat([h, hs[hop]], dim=-1))
            elif self.hop_aware_rectify_fn == "gru":
                h = h.contiguous().view(batch_size * n_series, 1, -1)  # (B * N, L, C'), L = 1
                if hs is None:
                    # Initial hidden states are zeros for the first layer
                    _, h = self.hop_aware_rectifiers[hop](h)  # (1, B * N, C')
                else:
                    _, h = self.hop_aware_rectifiers[hop](h, hs[hop])  # (1, B * N, C')
            elif self.hop_aware_rectify_fn == "glu":
                if hs is not None:
                    h = h + self.hop_aware_rectifiers[hop](hs[hop])
            elif self.hop_aware_rectify_fn == "mean":
                if hs is not None:
                    h = (h + hs[hop]) / 2

            h_latent.append(h)  # Fed into the next HARDPurG layer to facilitate rectifying

            h = h.contiguous().view(batch_size, n_series, -1)  # (B, N, C')
            h_mix = torch.cat((h_mix, h), dim=-1)  # Concat along channel

        # Information selection
        h = self.l_slc(h_mix)

        return h, h_latent
