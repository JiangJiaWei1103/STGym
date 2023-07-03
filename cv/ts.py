"""
Time series validation schemes.
Author: JiaWei Jiang

This file contains customized time series validators, splitting dataset
following chronological ordering.
"""
import math
from typing import Iterator, Tuple

import numpy as np
import pandas as pd


class TSSplit(object):
    """Data splitter using the naive train/val split scheme.

    Holdout set is reserved before data splitting, so there's no split
    for test set here.

    Parameters:
        train_ratio: ratio of training samples
        val_ratio: ratio of validation samples
    """

    def __init__(self, train_ratio: float, val_ratio: float):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

    def split(self, X: pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Return indices of training and validation sets.

        Because this is the implementation of naive train/val split,
        returned Iterator is the pseudo one. That is, only a single
        fold is considered to cater to the common experimental setting.

        Parameters:
            X: raw DataFrame

        Yield:
            tr_idx: training set indices for current split
            val_idx: validation set indices for current split
        """
        n_samples = len(X)
        train_end = math.floor(n_samples * self.train_ratio)
        val_end = train_end + math.floor(n_samples * self.val_ratio)

        tr_idx = np.arange(0, train_end)
        val_idx = np.arange(train_end, val_end)

        yield tr_idx, val_idx
