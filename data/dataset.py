"""
Dataset definitions.
Author: JiaWei Jiang

This file contains definitions of multiple datasets used in different
scenarios.

* [ ] Define a `BaseDataset`.
"""
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from pandas.api.types import is_datetime64_any_dtype
from torch import Tensor
from torch.utils.data import Dataset

from metadata import N_DAYS_IN_WEEK


class BenchmarkDataset(Dataset):
    """Benchmark Dataset for open source datasets in MTSF domain.

    Denote M as the number of total samples.

    Parameters:
        df: processed data
        t_window: lookback time window, denoted by T
        horizon: predicting horizon, denoted by Q
        n_tids: number of time slots per day
        add_diw: whether to add day in week as auxiliary feature
        name: name of the specified dataset, for compatibility
    """

    tids: Optional[np.ndarray] = None
    diws: Optional[np.ndarray] = None

    def __init__(
        self,
        df: pd.DataFrame,
        t_window: int,
        horizon: int,
        n_tids: Optional[int] = None,
        add_diw: bool = False,
        name: Optional[str] = None,
        **kwargs: Any,
    ):
        self.df = df
        self.t_window = t_window
        self.horizon = horizon
        self.n_tids = n_tids
        self.add_diw = add_diw

        self._set_n_samples()
        self._add_aux_feats()
        self._chunk_X_y()

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        # X, y = self._get_windowed_sample(idx)
        data_sample = {
            "X": torch.tensor(self.X[idx], dtype=torch.float32),
            "y": torch.tensor(self.y[idx], dtype=torch.float32),
        }

        cur_time_slot = idx + self.t_window - 1
        if self.n_tids is not None:
            data_sample["tid"] = torch.tensor(self.tids[cur_time_slot], dtype=torch.int32)
        if self.add_diw:
            data_sample["diw"] = torch.tensor(self.diws[cur_time_slot], dtype=torch.int32)

        return data_sample

    def _set_n_samples(self) -> None:
        """Set number of samples."""
        self.offset = self.t_window + self.horizon - 1
        self.n_samples = len(self.df) - self.offset

    def _add_aux_feats(self) -> None:
        """Add auxiliary features (i.e., time identifiers)."""
        # Add time in day
        if self.n_tids is not None:
            time_vals = self.df.index.values
            if is_datetime64_any_dtype(self.df.index):
                self.tids = (time_vals - time_vals.astype("datetime64[D]")) / np.timedelta64(1, "D")
                tids_uniq = sorted(np.unique(self.tids))
                self.tids = (self.tids / tids_uniq[1]).astype(np.int32)
            else:
                self.tids = (time_vals % self.n_tids).astype(np.int32)

        # Add day in week
        if self.add_diw:
            if is_datetime64_any_dtype(self.df.index):
                self.diws = self.df.index.dayofweek.values
            else:
                self.diws = (self.df.index.values // self.n_tids % N_DAYS_IN_WEEK).astype(np.int32)

    def _chunk_X_y(self) -> None:
        """Chunk X and y sets based on T and Q."""
        X = []
        y = []

        for i in range(self.n_samples):
            X.append(self.df.values[i : i + self.t_window, :])
            y.append(self.df.values[i + self.offset, :])

        self.X = np.stack(X)  # (M, T, N)
        self.y = np.stack(y)  # (M, N), Q = 1 for single-horizon

    def _get_windowed_sample(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (X, y) sample based on idx passed into __getitem__.

        Parameters:
            idx: index of the sample to retrieve

        Return:
            X: X sample corresponding to the given index
            y: y sample corresponding to the given index
        """
        X = self.df.values[idx : idx + self.t_window, :]
        y = self.df.values[idx + self.offset, :]

        return X, y


class TrafficDataset(Dataset):
    """Traffic Dataset for open source traffic forecasting datasets.

    For traffic datasets, the problem is formulated as multi-horizon
    forecasting, and the default setting is 12 predicting horizons.
    That is, forecasting is done for one hour later in the future.

    Parameters:
        df: processed data
        t_window: lookback time window
        horizon: predicting horizon
        n_tids: number of time slots per day
        add_diw: whether to add day in week as auxiliary feature
        name: name of the specified dataset, for compatibility
    """

    tids: Optional[np.ndarray] = None
    diws: Optional[np.ndarray] = None

    def __init__(
        self,
        df: Union[pd.DataFrame, np.ndarray],
        t_window: int = 12,
        horizon: int = 12,
        n_tids: Optional[int] = None,
        add_diw: bool = False,
        name: Optional[str] = None,
    ):
        self.df = df
        self.t_window = t_window
        self.horizon = horizon
        self.n_tids = n_tids
        self.add_diw = add_diw

        self._set_n_samples()
        self._add_aux_feats()
        self._chunk_X_y()

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        # X, y = self._get_windowed_sample(idx)
        data_sample = {
            "X": torch.tensor(self.X[idx], dtype=torch.float32),
            "y": torch.tensor(self.y[idx], dtype=torch.float32),
        }

        cur_time_slot = idx + self.t_window - 1
        if self.n_tids is not None:
            data_sample["tid"] = torch.tensor(self.tids[cur_time_slot], dtype=torch.int32)
        if self.add_diw:
            data_sample["diw"] = torch.tensor(self.diws[cur_time_slot], dtype=torch.int32)

        return data_sample

    def _set_n_samples(self) -> None:
        """Set number of samples."""
        self.offset = self.t_window + self.horizon - 1
        self.n_samples = len(self.df) - self.offset

    def _add_aux_feats(self) -> None:
        """Add auxiliary features (i.e., time identifiers)."""
        # Add time in day
        if self.n_tids is not None:
            time_vals = self.df.index.values
            if is_datetime64_any_dtype(self.df.index):
                self.tids = (time_vals - time_vals.astype("datetime64[D]")) / np.timedelta64(1, "D")
                tids_uniq = sorted(np.unique(self.tids))
                self.tids = (self.tids / tids_uniq[1]).astype(np.int32)
            else:
                self.tids = (time_vals % self.n_tids).astype(np.int32)

        # Add day in week
        if self.add_diw:
            if is_datetime64_any_dtype(self.df.index):
                self.diws = self.df.index.dayofweek.values
            else:
                self.diws = (self.df.index.values // self.n_tids % N_DAYS_IN_WEEK).astype(np.int32)

    def _chunk_X_y(self) -> None:
        """Chunk X and y sets based on T and Q."""
        if isinstance(self.df, pd.DataFrame):
            data_vals = self.df.values
        else:
            data_vals = self.df

        X = []
        y = []
        for i in range(self.n_samples):
            X.append(data_vals[i : i + self.t_window, :])
            y.append(data_vals[i + self.t_window : i + self.offset + 1, :])

        self.X = np.stack(X)  # (M, T, N)
        self.y = np.stack(y)  # (M, Q, N)

    def _get_windowed_sample(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (X, y) sample based on idx passed into __getitem__.

        Different from single-step forecasting, y here for multi-step
        forecasting is a sequence.

        Parameters:
            idx: index of the sample to retrieve

        Return:
            X: X sample corresponding to the given index
            y: y sample corresponding to the given index
        """
        if isinstance(self.df, pd.DataFrame):
            data_vals = self.df.values
        else:
            data_vals = self.df

        X = data_vals[idx : idx + self.t_window, :]
        y = data_vals[idx + self.t_window : idx + self.offset + 1, :]

        return X, y
