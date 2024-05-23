"""
Max scaler.
Author: JiaWei Jiang

This file contains the definition of max scaler, which divides all
values in dataset by the maximum, either global (i.e., across entities)
or within each entity (e.g., a sensor).
"""
from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
import torch
from torch import Tensor


class MaxScaler(object):
    """Scale dataset values by dividing them by the maximum.

    The scaled value of a sample `x` is computed as:
        z = x / x_m

    where `x_m` is the maximum of all the samples and it's extracted
    independently from each feature if `local=True`. Or, `x_m` will be
    the maximum across all the features (e.g., time series, nodes).

    Args:
        local: whether maximum value is extracted within each series
            (i.e., nodes) or across all series (globally, see TPA-LSTM)

    Attributes:
        max_: ndarray, the maximum value for each feature, with shape
            (n_features, )
    """

    def __init__(self, local: bool = True):
        self._local = local

    def fit(self, X: Union[pd.DataFrame, np.ndarray]) -> MaxScaler:
        """Compute the maximums to be used for later scaling.

        Args:
            X: data used to calculate maximum for later scaling, with
                shape (n_samples, n_features)

        Returns:
            self: fitted scaler
        """
        n_features = X.shape[1]
        if self._local:
            self.max_ = np.max(np.abs(X), axis=0)
        else:
            self.max_ = np.full(shape=(n_features,), fill_value=np.max(np.abs(X)))

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Perform scaling by dividing values by maximum.

        Args:
            X: data to scale, with shape (n_samples, n_features)

        Returns:
            X_tr: transformed (scaled) array, with shape as input
        """
        if isinstance(X, pd.DataFrame):
            # Temporary workaround for np.ndarray in Union has no attr values
            X_tr = X.values / self.max_
        else:
            X_tr = X / self.max_

        return X_tr

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Compute the maximums and scale the data."""
        self.fit(X)

        return self.transform(X)

    def inverse_transform(self, X: Union[np.ndarray, Tensor]) -> Union[np.ndarray, Tensor]:
        """Undo scaling of input data.

        It's commonly used when users want to get the prediction and
        groundtruths at the original scale.

        Args:
            X: input data to be inversely transformed, with shape
                (n_sampels, n_features)

        Returns:
            Xt: inversely transformed data, with shape as input
        """
        if isinstance(X, Tensor):
            max_ = torch.tensor(self.max_, dtype=torch.float32).to(X.device)
            Xt = X * max_
        else:
            Xt = X * self.max_

        return Xt


class StandardScaler(object):
    """Standardize features by removing the mean and scaling to unit
    variance.

    The standard score of a sample `x` is calculated as:
        z = (x - u) / s

    where `u` is the mean of the training samples or zero if
    `with_mean=False`, and `s` is the standard deviation of the
    training samples or one if `with_std=False`.

    This implementation is used to avoid the situation that inversely-
    transformed values become non-zeros for those original zeros.

    Attributes:
        mean_: np.float64, the global mean value for each entry in the
            training set
        std_: np.float64, the global standard deviation for each entry
            in the training set
    """

    def __init__(self) -> None:
        pass

    def fit(self, X: Union[pd.DataFrame, np.ndarray]) -> StandardScaler:
        """Compute the mean and std to be used for later scaling.

        Args:
            X: ndarray or pd.DataFrame, data used to calculate mean and
                standard deviation for later scaling, with shape
                (n_samples, n_features)

        Returns:
            self: fitted scaler
        """
        self.mean_ = np.mean(X)
        self.std_ = np.std(X)

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Perform standardization by centering and scaling.

        Args:
            X: data to scale, with shape (n_samples, n_features)

        Returns:
            X_tr: transformed (scaled) array, with shape as input
        """
        if isinstance(X, pd.DataFrame):
            # Temporary workaround for np.ndarray in Union has no attr values
            X_tr = (X.values - self.mean_) / self.std_
        else:
            X_tr = (X - self.mean_) / self.std_

        return X_tr

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Compute the mean and std and scale the data."""
        self.fit(X)

        return self.transform(X)

    def inverse_transform(self, X: Union[np.ndarray, Tensor]) -> np.ndarray:
        """Scale back the data to the original representation.

        It's commonly used when users want to get the prediction and
        groundtruths at the original scale.

        Args:
            X: input data to be inversely transformed, with shape
                (n_samples, n_features)

        Returns:
            Xt: inversely transformed data, with shape as input
        """
        Xt = X * self.std_ + self.mean_

        return Xt


class MinMaxScaler(object):
    """Normalize features into range [-1, 1].

    The normalized score of a sample `x` is calculated as:
        z = 2 * [(x - x_min) / (x_max - x_min)] - 1

    Attributes:
        min_: np.float64, the global min value for each entry in the
            training set
        max_: np.float64, the global max value for each entry in the
            training set
        range_: np.float64, the global value range for each entry in
            the training set
    """

    def __init__(self) -> None:
        pass

    def fit(self, X: Union[pd.DataFrame, np.ndarray]) -> MinMaxScaler:
        """Compute the min and max to be used for later scaling.

        Args:
            X: data used to calculate min and max for later scaling

        Returns:
            self: fitted scaler
        """
        self.min_ = np.min(X)
        self.max_ = np.max(X)

        self.range_ = self.max_ - self.min_

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Perform normalization by scaling to range [-1, 1].

        Args:
            X: data to scale, with shape (n_samples, n_features)

        Returns:
            X_tr: transformed (scaled) array, with shape as input
        """
        if isinstance(X, pd.DataFrame):
            # Temporary workaround for np.ndarray in Union has no attr values
            X_tr = 2 * ((X.values - self.min_) / self.range_) - 1
        else:
            X_tr = 2 * ((X - self.min_) / self.range_) - 1

        return X_tr

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Compute the min and max and scale the data."""
        self.fit(X)

        return self.transform(X)

    def inverse_transform(self, X: Union[np.ndarray, Tensor]) -> np.ndarray:
        """Scale back the data to the original representation.

        It's commonly used when users want to get the prediction and
        groundtruths at the original scale.

        Args:
            X: input data to be inversely transformed, with shape
                (n_samples, n_features)

        Returns:
            Xt: inversely transformed data, with shape as input
        """
        Xt = (X + 1) / 2 * self.range_ + self.min_

        return Xt
