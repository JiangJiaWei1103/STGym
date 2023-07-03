"""
Dataloader building logic.
Author: JiaWei Jiang

This file contains the basic logic of building dataloaders for training
and evaluation processes.
"""
from typing import Any, Optional, Tuple, Type, Union

import pandas as pd
from torch.utils.data import DataLoader

from metadata import MTSFBerks, TrafBerks

from .dataset import BenchmarkDataset, TrafficDataset


def build_dataloaders(
    df_tr: pd.DataFrame, df_val: pd.DataFrame, batch_size: int, shuffle: bool, num_workers: int, **dataset_cfg: Any
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create and return train and validation data loaders.

    Parameters:
        df_tr: training data
        df_val: validation data
        batch_size: number of samples per batch
        shuffle: whether to shuffle samples every epoch
        num_workers: number of subprocesses used to load data
        dataset_cfg: hyperparameters of customized dataset

    Return:
        train_loader: training data loader
        val_loader: validation data loader
    """
    dataset: Union[Type[BenchmarkDataset], Type[TrafficDataset]] = None
    if dataset_cfg["name"] in MTSFBerks:
        dataset = BenchmarkDataset
        collate_fn = None
    elif dataset_cfg["name"] in TrafBerks:
        dataset = TrafficDataset
        collate_fn = None

    train_loader = DataLoader(
        dataset(df_tr, **dataset_cfg),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    if df_val is not None:
        val_loader = DataLoader(
            dataset(df_val, **dataset_cfg),
            batch_size=256,  # 1024 for solar
            shuffle=False,  # Hard-coded
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

        return train_loader, val_loader
    else:
        return train_loader, None
