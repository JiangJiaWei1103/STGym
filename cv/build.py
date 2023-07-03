"""
Cross-validator building logic.
Author: JiaWei Jiang

This file contains the basic logic of building cross-validator.
"""
from typing import Any

from .ts import TSSplit


def build_cv(**cv_cfg: Any) -> TSSplit:
    """Build and return cross-validator.

    *Note: Currently only support time series splitting with a single
    fold (i.e., the naive train/val split).

    Parameters:
        cv_cfg: hyperparameters of cross-validator

    Return:
        cv: cross-validator
    """
    cv_scheme = cv_cfg["scheme"]
    n_folds = cv_cfg["n_folds"]
    train_ratio = cv_cfg["train_ratio"]
    val_ratio = cv_cfg["val_ratio"]

    if cv_scheme == "tssplit":
        # Naive train/val split following chronological ordering
        cv = TSSplit(train_ratio, val_ratio)

    return cv
