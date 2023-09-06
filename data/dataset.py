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
        day_window: Optional[int] = None,
        week_window: Optional[int] = None,
        num_of_day: Optional[int] = None,
        num_of_week: Optional[int] = None,
        horizon: int = 12,
        n_tids: Optional[int] = None,
        add_tid: bool = False,
        add_diw: bool = False,
        name: Optional[str] = None,
    ):
        self.df = df
        self.t_window = t_window
        self.day_window = day_window
        self.week_window = week_window
        self.num_of_day = num_of_day
        self.num_of_week = num_of_week
        self.horizon = horizon
        self.n_tids = n_tids
        self.add_tid = add_tid
        self.add_diw = add_diw

        self._set_n_samples()
        self._add_aux_feats()
        self._chunk_X_y()

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        # X, y = self._get_windowed_sample(idx)
        cur_time_slot = idx + self.t_window - 1

        X = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        X_tid = None
        y_tid = None
        X_diw = None
        y_diw = None

        if self.add_tid:
            X_tid = torch.tensor(self.tids[cur_time_slot], dtype=torch.int32)
            y_tid = torch.tensor(self.tids[cur_time_slot + self.horizon], dtype=torch.int32)
        if self.add_diw:
            X_diw = torch.tensor(self.diws[cur_time_slot], dtype=torch.int32)
            y_diw = torch.tensor(self.diws[cur_time_slot + self.horizon], dtype=torch.int32)

        if self.add_tid or self.add_diw:
            X = self._add_aux_info(X.squeeze(0).transpose(1, 0), X_tid, X_diw)
            y = self._add_aux_info(y.squeeze(0).transpose(1, 0), y_tid, y_diw)

        if (self.num_of_day is None) and (self.num_of_week is None):
            data_sample = {
                "X": X,
                "y": y,
            }
        elif (self.num_of_day is not None) and (self.num_of_week is None):
            data_sample = {
                "X": X,
                "X_day": torch.tensor(self.X_day[idx], dtype=torch.float32),
                "y": y,
            }
        elif (self.num_of_day is None) and (self.num_of_week is not None):
            data_sample = {
                "X": X,
                "X_week": torch.tensor(self.X_week[idx], dtype=torch.float32),
                "y": y,
            }
        else:
            data_sample = {
                "X": X,
                "X_day": torch.tensor(self.X_day[idx], dtype=torch.float32),
                "X_week": torch.tensor(self.X_week[idx], dtype=torch.float32),
                "y": y,
            }

        return data_sample

    def _set_n_samples(self) -> None:
        """Set number of samples."""
        if (self.num_of_day is None) and (self.num_of_week is None):
            self.offset = self.t_window + self.horizon - 1
            self.n_samples = len(self.df) - self.offset
        elif (self.num_of_day is not None) and (self.num_of_week is None):
            self.offset = self.n_tids * self.num_of_day + self.horizon - 1
            self.n_samples = len(self.df) - self.offset
        else:
            self.offset = self.n_tids * 7 * self.num_of_week + self.horizon - 1
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

    def _add_aux_info(
        self,
        x: Tensor,
        tid: Tensor = None,
        diw: Tensor = None
    ) -> Tensor:
        """
        Add auxiliary information along feature (i.e., channel) axis.

        It's used in traffic forecasting scenarios, where time identifiers
        are considered to be auxiliary information.
        """

        N_TIMES_IN_DAY = self.n_tids
        N_DAYS_IN_WEEK = 7
        self.lookback_idx = torch.arange(self.t_window)

        t_window, n_series = x.shape
        x = torch.unsqueeze(x.transpose(0, 1), dim=0)  # (C, N, T)

        if self.lookback_idx.device != x.device:
            self.lookback_idx = self.lookback_idx.to(x.device)

        if tid is not None:
            tid_expand = tid.unsqueeze(dim=-1).unsqueeze(dim=-1).expand(n_series, t_window)  # (N, T)
            tid_back = (((tid_expand + N_TIMES_IN_DAY) - self.lookback_idx) % N_TIMES_IN_DAY).flip(
                dims=[1]
            ) / N_TIMES_IN_DAY
            x = torch.cat([x, tid_back.unsqueeze(dim=0)], dim=0)
        if diw is not None:
            diw_expand = diw.unsqueeze(dim=-1).unsqueeze(dim=-1).expand(n_series, t_window)  # (N, T)
            cross_day_mask = ((tid_expand - self.lookback_idx).flip(dims=[1]) < 0).int()
            diw_back = (
                ((diw_expand + N_DAYS_IN_WEEK) - cross_day_mask) % N_DAYS_IN_WEEK / N_DAYS_IN_WEEK
            )
            x = torch.cat([x, diw_back.unsqueeze(dim=0)], dim=0)

        return x

    def _chunk_X_y(self) -> None:
        """Chunk X and y sets based on T and Q."""
        if isinstance(self.df, pd.DataFrame):
            data_vals = self.df.values
        else:
            data_vals = self.df
        
        if (self.num_of_day is None) and (self.num_of_week is None):
            X = []
            y = []
            for i in range(self.n_samples):
                X.append(data_vals[i : i + self.t_window, :])
                y.append(data_vals[i + self.t_window : i + self.offset + 1, :])

            self.X = np.expand_dims(np.stack(X).transpose(0 ,2, 1), 1)  # (M, 1, N, T)
            self.y = np.expand_dims(np.stack(y).transpose(0 ,2, 1), 1)  # (M, 1, Q, N)

        elif (self.num_of_day is not None) and (self.num_of_week is None):
            X = []
            X_day = []
            y = []
            for i in range(self.n_samples):
                start = self.n_tids * self.num_of_day - self.t_window + i
                day = np.empty((0, data_vals.shape[1]))
                for j in range(self.num_of_day):
                    day = np.concatenate(
                        [day, 
                         data_vals[i + j * self.n_tids : 
                                   i + j * self.n_tids + self.day_window, :]])
                X.append(data_vals[start : start + self.t_window, :])
                X_day.append(day)
                y.append(data_vals[start + self.t_window : i + self.offset + 1, :])

            self.X = np.expand_dims(np.stack(X).transpose(0 ,2, 1), 1)  # (M, 1, N, T)
            self.X_day = np.expand_dims(np.stack(X_day).transpose(0 ,2, 1), 1)  # (M, 1, N, T_day)
            self.y = np.expand_dims(np.stack(y).transpose(0 ,2, 1), 1)  # (M, 1, Q, N)

        elif (self.num_of_day is None) and (self.num_of_week is not None):
            X = []
            X_week = []
            y = []
            for i in range(self.n_samples):
                start = self.n_tids * 7 * self.num_of_week - self.t_window + i
                week = np.empty((0, data_vals.shape[1]))
                for j in range(self.num_of_week):
                    week = np.concatenate(
                        [week, 
                         data_vals[i + j * self.n_tids * 7 : 
                                   i + j * self.n_tids * 7 + self.week_window, :]])
                X.append(data_vals[start : start + self.t_window, :])
                X_week.append(week)
                y.append(data_vals[start + self.t_window : i + self.offset + 1, :])
            
            self.X = np.expand_dims(np.stack(X).transpose(0 ,2, 1), 1)  # (M, 1, N, T)
            self.X_week = np.expand_dims(np.stack(X_week).transpose(0 ,2, 1), 1)  # (M, 1, N, T_week)
            self.y = np.expand_dims(np.stack(y).transpose(0 ,2, 1), 1)  # (M, 1, Q, N)

        else:
            X = []
            X_day = []
            X_week = []
            y = []
            for i in range(self.n_samples):
                start = self.n_tids * 7 * self.num_of_week - self.t_window + i
                start_day = start + self.t_window - self.num_of_day * self.n_tids
                day = np.empty((0, data_vals.shape[1]))
                for j in range(self.num_of_day):
                    day = np.concatenate(
                        [day, 
                         data_vals[start_day + j * self.n_tids : 
                                   start_day + j * self.n_tids + self.day_window, :]])
                week = np.empty((0, data_vals.shape[1]))
                for k in range(self.num_of_week):
                    week = np.concatenate(
                        [week, 
                         data_vals[i + k * self.n_tids * 7 : 
                                   i + k * self.n_tids * 7 + self.week_window, :]])
                X.append(data_vals[start : start + self.t_window, :])
                X_day.append(day)
                X_week.append(week)
                y.append(data_vals[start + self.t_window : i + self.offset + 1, :])

            self.X = np.expand_dims(np.stack(X).transpose(0 ,2, 1), 1)  # (M, 1, N, T)
            self.X_day = np.expand_dims(np.stack(X_day).transpose(0 ,2, 1), 1)  # (M, 1, N, T_day)
            self.X_week = np.expand_dims(np.stack(X_week).transpose(0 ,2, 1), 1)  # (M, 1, N, T_week)
            self.y = np.expand_dims(np.stack(y).transpose(0 ,2, 1), 1)  # (M, 1, Q, N)

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
