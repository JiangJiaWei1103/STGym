"""
Time series cross-validation schemes.
Author: JiaWei Jiang

This file contains definitions of time series cross-validation schemes
which splits dataset following chronological ordering.
"""
import math
from typing import Iterator, Tuple

import numpy as np


class TSSplit(object):
    """Time series cross-validation scheme with train/val split.

    Args:
        train_ratio: Ratio of training samples.
        val_ratio: Ratio of validation samples.
    """

    def __init__(self, train_ratio: float, val_ratio: float) -> None:
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

    def split(self, X: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Returns indices of training and validation sets.

        Args:
            X: Input data.

        Yields:
            A tuple (tr_idx, val_idx), where tr_idx is the training
            index array and val_idx is the validation index array.
        """
        n_samples = len(X)
        train_end = math.floor(n_samples * self.train_ratio)
        val_end = train_end + math.floor(n_samples * self.val_ratio)

        tr_idx = np.arange(0, train_end)
        val_idx = np.arange(train_end, val_end)

        yield tr_idx, val_idx
