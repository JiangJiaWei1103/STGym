"""
Dataset definitions.
Author: JiaWei Jiang

This file contains definitions of multiple datasets used in different
scenarios.

* [ ] Define a `BaseDataset` (e.g., containing common dataset attrs).
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, Optional, Union

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
    * P_day: days lookback window
    * P_week: weeks lookback window
    * n_day: number of days for lookback window
    * n_week: number of weeks for lookback window
    * n_tids: number of time slots in one day
    """

    M: int
    N: int
    P: int
    Q: int
    P_day: Union[int, None]
    P_week: Union[int, None]
    n_day: Union[int, None]
    n_week: Union[int, None]
    n_tids: Union[int, None]
    offset: int = field(init=False)
    n_samples: int = field(init=False)

    def __post_init__(self) -> None:
        if (self.n_day is None) and (self.n_week is None):
            self.offset = self.P + self.Q - 1
        elif (self.n_day is not None) and (self.n_week is None):
            self.offset = self.n_tids * self.n_day + self.Q - 1
        else:
            self.offset = self.n_tids * 7 * self.n_week + self.Q - 1    
        
        self.n_samples = self.M - self.offset

class BenchmarkDataset(Dataset):
    """Benchmark Dataset for open source datasets in MTSF domain.

    Data contains only time series numeric values.

    Parameters:
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
        """Return (X, y) sample based on idx passed into __getitem__.

        Parameters:
            idx: index of the sample to retrieve

        Return:
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

    Parameters:
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
        n_tids: Optional[int] = None,
        day_window: Optional[int] = None,
        week_window: Optional[int] = None,
        num_of_day: Optional[int] = None,
        num_of_week: Optional[int] = None, 
        **kwargs: Any,
    ):
        self.data = data
        self.t_window = t_window
        self.day_window = day_window
        self.week_window = week_window
        self.num_of_day = num_of_day
        self.num_of_week = num_of_week
        self.horizon = horizon
        self.n_tids = n_tids

        self.ts_attr = _TimeSeriesAttr(
            len(data),
            data.shape[1],
            t_window,
            horizon,
            day_window,
            week_window,
            num_of_day,
            num_of_week,
            n_tids)

        self._chunk_X_y()

    def __len__(self) -> int:
        return self.ts_attr.n_samples

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        # X, y = self._get_windowed_sample(idx)

        data_sample = {
            "X": torch.tensor(self.X[idx], dtype=torch.float32),
            "y": torch.tensor(self.y[idx], dtype=torch.float32),
        }

        try:
            data_sample["X_day"] = torch.tensor(self.X_day[idx], dtype=torch.float32)
        except: 
            pass

        try:
            data_sample["X_week"] = torch.tensor(self.X_week[idx], dtype=torch.float32)
        except: 
            pass

        return data_sample

    def _chunk_X_y(self) -> None:
        """Chunk X and y sets based on T and Q."""
        
        if (self.ts_attr.n_day is None) and (self.ts_attr.n_week is None):
            X = []
            y = []
            for i in range(self.ts_attr.n_samples):
                X.append(self.data[i : i + self.ts_attr.P, ...])
                y.append(self.data[i + self.ts_attr.P : i + self.ts_attr.offset + 1, ...])

            self.X = np.stack(X)  # (M, P, N, C)
            self.y = np.stack(y)  # (M, Q, N, C)

        elif (self.ts_attr.n_day is not None) and (self.ts_attr.n_week is None):
            X = []
            y = []
            X_day = []
            for i in range(self.ts_attr.n_samples):
                start = self.ts_attr.n_tids * self.ts_attr.n_day - self.ts_attr.P + i
                day = np.empty((0, self.data.shape[1], self.data.shape[2]))
                for j in range(self.ts_attr.n_day):
                    day = np.concatenate(
                        [day, 
                         self.data[i + j * self.ts_attr.n_tids : 
                                   i + j * self.ts_attr.n_tids + self.ts_attr.P_day, ...]])
                X.append(self.data[start : start + self.ts_attr.P, ...])
                y.append(self.data[start + self.ts_attr.P : i + self.ts_attr.offset + 1, ...])
                X_day.append(day)

            self.X = np.stack(X)  # (M, P, N, C)
            self.y = np.stack(y)  # (M, Q, N, C)
            self.X_day = np.stack(X_day)  # (M, P_day, N, C)

        elif (self.ts_attr.n_day is None) and (self.ts_attr.n_week is not None):
            X = []
            y = []
            X_week = []
            for i in range(self.ts_attr.n_samples):
                start = self.ts_attr.n_tids * 7 * self.ts_attr.n_week - self.ts_attr.P + i
                week = np.empty((0, self.data.shape[1], self.data.shape[2]))
                for j in range(self.ts_attr.n_week):
                    week = np.concatenate(
                        [week, 
                         self.data[i + j * self.ts_attr.n_tids * 7 : 
                                   i + j * self.ts_attr.n_tids * 7 + self.ts_attr.P_week, ...]])
                X.append(self.data[start : start + self.ts_attr.P, ...])
                y.append(self.data[start + self.ts_attr.P : i + self.ts_attr.offset + 1, ...])
                X_week.append(week)
            
            self.X = np.stack(X)  # (M, P, N, C)
            self.y = np.stack(y)  # (M, Q, N, C)
            self.X_week = np.stack(X_week)  # (M, P_week, N, C)

        else:
            X = []
            y = []
            X_day = []
            X_week = []
            for i in range(self.ts_attr.n_samples):
                start = self.ts_attr.n_tids * 7 * self.ts_attr.n_week - self.ts_attr.P + i
                start_day = start + self.ts_attr.P - self.ts_attr.n_day * self.ts_attr.n_tids
                day = np.empty((0, self.data.shape[1], self.data.shape[2]))
                for j in range(self.ts_attr.n_day):
                    day = np.concatenate(
                        [day, 
                         self.data[start_day + j * self.ts_attr.n_tids : 
                                   start_day + j * self.ts_attr.n_tids + self.ts_attr.P_day, ...]])
                week = np.empty((0, self.data.shape[1], self.data.shape[2]))
                for k in range(self.ts_attr.n_week):
                    week = np.concatenate(
                        [week, 
                         self.data[i + k * self.ts_attr.n_tids * 7 : 
                                   i + k * self.ts_attr.n_tids * 7 + self.ts_attr.P_week, ...]])
                X.append(self.data[start : start + self.ts_attr.P, ...])
                y.append(self.data[start + self.ts_attr.P : i + self.ts_attr.offset + 1, ...])
                X_day.append(day)
                X_week.append(week)

            self.X = np.stack(X)  # (M, P, N, C)
            self.y = np.stack(y)  # (M, Q, N, C)
            self.X_day = np.stack(X_day)  # (M, P_day, N, C)
            self.X_week = np.stack(X_week)  # (M, P_week, N, C)

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
        if isinstance(self.data, pd.DataFrame):
            data_vals = self.data.values
        else:
            data_vals = self.data

        X = data_vals[idx : idx + self.t_window, :]
        y = data_vals[idx + self.t_window : idx + self.offset + 1, :]

        return X, y
