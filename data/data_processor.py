"""
Data processor definitions.
Author: JiaWei Jiang

This file contains the definition of data processor cleaning and
processing raw data before entering modeling phase.
"""
import logging
import math
import os
import pickle
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from pandas.api.types import is_datetime64_any_dtype
from scipy.sparse import coo_matrix, csr_matrix
from torch import Tensor

from metadata import N_DAYS_IN_WEEK, N_SERIES, MTSFBerks, TrafBerks
from paths import RAW_DATA_PATH
from utils.common import asym_norm, calculate_random_walk_matrix, calculate_scaled_laplacian, sym_norm
from utils.scaler import MaxScaler, MinMaxScaler, StandardScaler


class DataProcessor(object):
    """Data processor processing raw data, and providing access to
    processed data ready to be fed into modeling phase.

    Parameters:
       file_path: path of the raw data
           *Note: File reading supports .parquet extension in default
               setting, which can be modified to customized one.
        dataset_name: name of the dataset
        dp_cfg: hyperparameters of data processor
    """

    # https://stackoverflow.com/questions/59173744
    _data: np.ndarray
    _data_cv: np.ndarray
    _data_holdout: np.ndarray
    _data_test: np.ndarray

    # The priori graph structure is optionally provided
    _priori_adj_mat: Optional[List[Union[Tensor, coo_matrix, csr_matrix]]] = None

    def __init__(self, file_path: str, dataset_name: str, **dp_cfg: Any):
        self.file_path = file_path
        self.dataset_name = dataset_name

        # Setup data processor
        self._dp_cfg = dp_cfg
        self._setup()

        # Load raw data
        self._load_data()

    def _setup(self) -> None:
        """Retrieve hyperparameters for data processing."""
        # Before data splitting
        self.time_enc_params = self._dp_cfg["time_enc"]
        self.holdout_ratio = self._dp_cfg["holdout_ratio"]

        # After data splitting
        self.scaling = self._dp_cfg["scaling"]
        self.priori_gs = self._dp_cfg["priori_gs"]

        # Common
        self.n_series = N_SERIES[self.dataset_name]

    def _load_data(self) -> None:
        """Load raw data."""
        logging.info("Load data...")
        if self.dataset_name in MTSFBerks:
            data_vals = np.loadtxt(self.file_path, delimiter=",")
            self._df = pd.DataFrame(data_vals)
        elif self.dataset_name in TrafBerks:
            if self.file_path.endswith("npz"):
                data_vals = np.load(self.file_path, allow_pickle=True)["data"][..., 0]
                self._df = pd.DataFrame(data_vals)
            else:
                self._df = pd.read_hdf(self.file_path)

        logging.info(f"\t>> Data shape: {self._df.shape}")
        assert self._df.shape[1] == self.n_series, "#Series doesn't match."

    def run_before_splitting(self) -> None:
        """Clean and process data before data splitting (i.e., on raw
        static DataFrame).

        Return:
            None
        """
        logging.info("Run data cleaning and processing before data splitting...")

        # Concatenate time stamp encoding
        self._data = self._add_time_stamp_encoding(self._df)

        # Holdout unseen test set
        if self.holdout_ratio != 0:
            self._holdout()

        # Initialize priori graph structure if provided
        if self.priori_gs["type"] is not None:
            self._init_priori_gs()

    def run_after_splitting(self, data_tr: np.ndarray, data_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Any]:
        """Clean and process data after data splitting to avoid data
        leakage issue.

        Note that data processing is prone to data leakage, such as
        fitting the scaler with the whole dataset (including training).

        Parameters:
            data_tr: training data
            data_val: validation data

        Return:
            data_tr: processed training data
            data_val: processed validation data
            scaler: scaling object
        """
        logging.info("Run data cleaning and processing after data splitting...")
        scaler = None
        if self.scaling is not None:
            data_tr, data_val, scaler = self._scale(data_tr, data_val)

        return data_tr, data_val, scaler

    def get_df(self) -> pd.DataFrame:
        """Return raw DataFrame."""
        return self._df

    def get_data_cv(self) -> np.ndarray:
        """Return data for CV iteration."""
        return self._data_cv

    def get_data_test(self) -> np.ndarray:
        """Return unseen test set for final evaluation."""
        return self._data_test

    def get_priori_gs(self) -> Optional[List[Tensor]]:
        """Return priori graph structure."""
        return self._priori_adj_mat

    def _add_time_stamp_encoding(self, df: pd.DataFrame) -> np.ndarray:
        """Concatenate time stamp encoding to time series matrix."""
        logging.info("\t>> Add time stamp encoding...")
        time_enc = _TimeEncoder(**self.time_enc_params)
        data = time_enc.encode(df)

        return data

    def _holdout(self) -> None:
        """Holdout unseen test set before running CV iteration.

        `self._data_holdout` can prevent `self._data_test` from being
        modified repeatedly.
        """
        logging.info(f"\t>> Holdout unseen test with ratio {self.holdout_ratio}...")
        holdout_size = math.floor(len(self._data) * self.holdout_ratio)
        cv_size = len(self._data) - holdout_size

        self._data_holdout = self._data[-holdout_size:, ...]
        self._data_cv = self._data[:cv_size, ...]
        logging.info(f"\t#CV time steps: {len(self._data_cv)} | #Holdout time steps: {len(self._data_holdout)}")

    def _init_priori_gs(self) -> None:
        """Initialize the priori graph structure.

        See https://github.com/nnzhan/Graph-WaveNet/ .

        Return:
            None
        """
        logging.info(f"\t>> Initialize pre-defined graph structure with type {self.priori_gs['type']}...")
        priori_gs_type = self.priori_gs["type"]

        if priori_gs_type == "identity":
            self._priori_adj_mat = [torch.eye(self.n_series)]
        else:
            adj_mat = self._load_adj_mat()
            assert self.n_series == adj_mat.shape[0], "Shape of the adjacency matrix is wrong."

            if priori_gs_type == "sym_norm":
                self._priori_adj_mat = [sym_norm(adj_mat)]
            elif priori_gs_type == "transition":
                self._priori_adj_mat = [asym_norm(adj_mat)]
            elif priori_gs_type == "dbl_transition":
                self._priori_adj_mat = [asym_norm(adj_mat), asym_norm(adj_mat.T)]
            elif priori_gs_type == "laplacian":
                self._priori_adj_mat = [calculate_scaled_laplacian(adj_mat, lambda_max=None)]
            elif priori_gs_type == "random_walk":
                self._priori_adj_mat = [calculate_random_walk_matrix(adj_mat).T]
            elif priori_gs_type == "dual_random_walk":
                self._priori_adj_mat = [
                    calculate_random_walk_matrix(adj_mat).T,
                    calculate_random_walk_matrix(adj_mat.T).T,
                ]
            else:
                raise RuntimeError(f"Priori GS {priori_gs_type} isn't registered.")

        # ===
        # To torch sparse?
        def _build_sparse_matrix(L):  # type: ignore
            shape = L.shape
            i = torch.LongTensor(np.vstack((L.row, L.col)).astype(int))
            v = torch.FloatTensor(L.data)
            return torch.sparse.FloatTensor(i, v, torch.Size(shape))

        self._priori_adj_mat = [_build_sparse_matrix(A) for A in self._priori_adj_mat]
        # ===

    def _load_adj_mat(self) -> np.ndarray:
        """Load hand-crafted adjacency matrix.

        See https://github.com/nnzhan/Graph-WaveNet/ .

        Return:
            adj_mat: hand-crafted (pre-defined) adjacency matrix
        """
        adj_mat_file_path = os.path.join(RAW_DATA_PATH, self.dataset_name, f"{self.dataset_name}_adj.pkl")

        try:
            with open(adj_mat_file_path, "rb") as f:
                adj_mat = pickle.load(f)
        except UnicodeDecodeError as e:
            with open(adj_mat_file_path, "rb") as f:
                *_, adj_mat = pickle.load(f, encoding="latin1")
        except Exception as e:
            logging.error("Fail to load the hand-crafted adjacency matrix...")
            logging.error("Err:", e)
            raise

        return adj_mat

    def _scale(
        self,
        data_tr: np.ndarray,
        data_val: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Any]:
        """Scale the data.

        Parameters:
            data_tr: training data
            data_val: validation data

        Return:
            data_tr: scaled training data
            data_val: scaled validation data
            scaler: scaling object
        """
        logging.info(f"\t>> Scale data using {self.scaling} scaler...")

        scaler: Union[StandardScaler, MinMaxScaler, MaxScaler]
        if self.scaling == "standard":
            scaler = StandardScaler()
        elif self.scaling == "minmax":
            scaler = MinMaxScaler()
        elif self.scaling == "max":
            # See LSTNet
            scaler = MaxScaler()

        # Scale data
        data_tr[..., 0] = scaler.fit_transform(data_tr[..., 0])
        data_val[..., 0] = scaler.transform(data_val[..., 0])
        if self._data_holdout is not None:
            # Scale holdout test set
            self._data_test = self._data_holdout.copy()
            self._data_test[..., 0] = scaler.transform(self._data_test[..., 0])

        return data_tr, data_val, scaler


class _TimeEncoder(object):
    """Time encoder concatenating tiem stamp encodings with different
    granularities to time series matrix (numeric values).

    Commonly used notations are defined as follows:
    * M: number of rows in raw data
    * N: number of time series
    * C: number of channels (i.e., features)

    Parameters:
        add_tid: if True, add "Time in day" identifier as the auxiliary
            feature
        add_diw: if True, add "Day in week" identifier as the auxiliary
            feature
        n_tids: number of time slots in one day
        max_norm: if True, encoding is normalized by the max code
    """

    # Base number for max normalization
    tid_base: Optional[int]
    diw_base: Optional[int]

    def __init__(
        self, add_tid: bool = True, add_diw: bool = False, n_tids: Optional[int] = None, max_norm: bool = False
    ) -> None:
        self.add_tid = add_tid
        self.add_diw = add_diw
        self.n_tids = n_tids
        self.max_norm = max_norm

        self._setup()

    def _setup(self) -> None:
        """Setup time encoder."""
        if self.add_tid or self.add_diw:
            assert self.n_tids is not None, "Please provide `n_tids` for `TimeEncoder`."

        if self.max_norm:
            if self.add_tid:
                self.tid_base = self.n_tids
            if self.add_diw:
                self.diw_base = N_DAYS_IN_WEEK

    def encode(self, x: pd.DataFrame) -> np.ndarray:
        """Concatenate time stamp encodings to input time series.

        Time stamp encodings are concatenated along the channel dim.

        Parameters:
            x: input time series matrix

        Return:
            inputs: input time series matrix with time stamp encodings

        Shape:
            x: (M, N)
            inputs: (M, N, C)
        """
        inputs = [np.expand_dims(x.values, axis=-1)]
        n_series = x.shape[1]

        time_idx = x.index
        if self.add_tid:
            tids = np.tile(self._encode_tid(time_idx), [1, n_series, 1]).transpose((2, 1, 0))
            inputs.append(tids)
        if self.add_diw:
            diws = np.tile(self._encode_diw(time_idx), [1, n_series, 1]).transpose((2, 1, 0))
            inputs.append(diws)
        inputs = np.concatenate(inputs, axis=-1)

        return inputs

    def _encode_tid(self, time_idx: pd.Index) -> np.ndarray:
        """Generate and return time-in-day encoding."""
        time_vals = time_idx.values
        if is_datetime64_any_dtype(time_idx):
            tids = (time_vals - time_vals.astype("datetime64[D]")) / np.timedelta64(1, "D")
            tids_uniq = sorted(np.unique(tids))
            tids = (tids / tids_uniq[1]).astype(np.int32)
        else:
            tids = (time_vals % self.n_tids).astype(np.int32)
        if self.max_norm:
            tids = tids / self.tid_base

        return tids

    def _encode_diw(self, time_idx: pd.Index) -> np.ndarray:
        """Generate and return day-in-week encoding."""
        time_vals = time_idx.values
        if is_datetime64_any_dtype(time_idx):
            diws = time_idx.dayofweek.values
        else:
            diws = (time_vals // self.n_tids % N_DAYS_IN_WEEK).astype(np.int32)
        if self.max_norm:
            diws = diws / self.diw_base

        return diws
