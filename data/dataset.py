"""
Dataset definitions.
Author: JiaWei Jiang

This file contains definitions of multiple datasets used in different
scenarios.

* [ ] Define a `BaseDataset` (e.g., containing common dataset attrs).
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass
class _TimeSeriesAttr:
    """Common time series attributes.

    Commonly used notations are defined as follows:
    * M: number of rows in processed data
    * N: number of time series
    * P: lookback time window
    * Q: predicting horizon
    """

    M: int
    N: int
    P: int
    Q: int
    offset: int = field(init=False)
    n_samples: int = field(init=False)

    def __post_init__(self) -> None:
        self.offset = self.P + self.Q - 1
        self.n_samples = self.M - self.offset


class BenchmarkDataset(Dataset):
    """Benchmark Dataset for open source datasets in MTSF domain.

    Data contains only time series numeric values.

    Args:
        data: processed data
        t_window: lookback time window, denoted by T
        horizon: predicting horizon, denoted by Q
    """

    def __init__(
        self,
        data: np.ndarray,
        t_window: int,
        horizon: int,
        **kwargs: Any,
    ):
        self.data = data
        self.ts_attr = _TimeSeriesAttr(len(data), data.shape[1], t_window, horizon)

        self._chunk_X_y()

    def __len__(self) -> int:
        return self.ts_attr.n_samples

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        # X, y = self._get_windowed_sample(idx)
        data_sample = {
            "X": torch.tensor(self.X[idx], dtype=torch.float32),
            "y": torch.tensor(self.y[idx], dtype=torch.float32),
        }

        # cur_time_slot = idx + self.t_window - 1
        # if self.n_tids is not None:
        #    data_sample["tid"] = torch.tensor(self.tids[cur_time_slot], dtype=torch.int32)
        # if self.add_diw:
        #    data_sample["diw"] = torch.tensor(self.diws[cur_time_slot], dtype=torch.int32)

        return data_sample

    def _chunk_X_y(self) -> None:
        """Chunk X and y sets based on lookback and horizon."""
        X = []
        y = []

        for i in range(self.ts_attr.n_samples):
            X.append(self.data[i : i + self.ts_attr.P, ...])
            y.append(self.data[i + self.ts_attr.offset, :, 0])

        self.X = np.stack(X)  # (M, P, N, C)
        self.y = np.stack(y)  # (M, N), Q = 1 for single-horizon

    def _get_windowed_sample(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (X, y) sample based on idx passed into __getitem__.

        Args:
            idx: index of the sample to retrieve

        Returns:
            X: X sample corresponding to the given index
            y: y sample corresponding to the given index
        """
        X = self.data.values[idx : idx + self.t_window, :]
        y = self.data.values[idx + self.offset, :]

        return X, y


class TrafficDataset(Dataset):
    """Traffic Dataset for open source traffic forecasting datasets.

    For traffic datasets, the problem is formulated as multi-horizon
    forecasting, and the default setting is 12 predicting horizons.
    That is, forecasting is done for one hour later in the future.

    Args:
        data: processed data
        t_window: lookback time window
        horizon: predicting horizon
        n_tids: number of time slots per day
        add_diw: whether to add day in week as auxiliary feature
        name: name of the specified dataset, for compatibility
    """

    def __init__(
        self,
        data: np.ndarray,
        t_window: int,
        horizon: int,
        **kwargs: Any,
    ):
        self.data = data
        self.ts_attr = _TimeSeriesAttr(len(data), data.shape[1], t_window, horizon)

        self._chunk_X_y()

    def __len__(self) -> int:
        return self.ts_attr.n_samples

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        # X, y = self._get_windowed_sample(idx)
        data_sample = {
            "X": torch.tensor(self.X[idx], dtype=torch.float32),
            "y": torch.tensor(self.y[idx], dtype=torch.float32),
        }

        # cur_time_slot = idx + self.t_window - 1
        # if self.n_tids is not None:
        #    data_sample["tid"] = torch.tensor(self.tids[cur_time_slot], dtype=torch.int32)
        # if self.add_diw:
        #   data_sample["diw"] = torch.tensor(self.diws[cur_time_slot], dtype=torch.int32)

        return data_sample

    def _chunk_X_y(self) -> None:
        """Chunk X and y sets based on lookback and horizon"""
        X = []
        y = []
        for i in range(self.ts_attr.n_samples):
            X.append(self.data[i : i + self.ts_attr.P, ...])
            y.append(self.data[i + self.ts_attr.P : i + self.ts_attr.offset + 1, ...])

        self.X = np.stack(X)  # (M, P, N, C)
        self.y = np.stack(y)  # (M, Q, N, C)

    def _get_windowed_sample(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (X, y) sample based on idx passed into __getitem__.

        Different from single-step forecasting, y here for multi-step
        forecasting is a sequence.

        Args:
            idx: index of the sample to retrieve

        Returns:
            X: X sample corresponding to the given index
            y: y sample corresponding to the given index
        """
        if isinstance(self.data, pd.DataFrame):
            data_vals = self.data.values
        else:
            data_vals = self.data

        X = data_vals[idx : idx + self.t_window, :]
        y = data_vals[idx + self.t_window : idx + self.offset + 1, :]

        return X, y
