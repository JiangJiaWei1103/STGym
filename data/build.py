"""
Dataloader building logic.
Author: JiaWei Jiang

This file contains the basic logic of building dataloaders for training
and evaluation processes.
"""
from typing import Any, Optional, Tuple, Type, Union

import numpy as np
from torch.utils.data import DataLoader

from metadata import MTSFBerks, TrafBerks

from .dataset import BenchmarkDataset, TrafficDataset


def build_dataloaders(
    dataset_name: str,
    data_tr: np.ndarray,
    data_val: np.ndrray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    **dataset_cfg: Any,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create and return train and validation data loaders.

    Parameters:
        dataset_name: name of the dataset
        data_tr: training data
        data_val: validation data
        batch_size: number of samples per batch
        shuffle: whether to shuffle samples every epoch
        num_workers: number of subprocesses used to load data
        dataset_cfg: hyperparameters of customized dataset

    Return:
        train_loader: training data loader
        val_loader: validation data loader
    """
    dataset: Union[Type[BenchmarkDataset], Type[TrafficDataset]] = None
    if dataset_name in MTSFBerks:
        dataset = BenchmarkDataset
        collate_fn = None
    elif dataset_name in TrafBerks:
        dataset = TrafficDataset
        collate_fn = None

    train_loader = DataLoader(
        dataset(data_tr, **dataset_cfg),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    if data_val is not None:
        val_loader = DataLoader(
            dataset(data_val, **dataset_cfg),
            batch_size=batch_size * 4,  # 1024 for solar
            shuffle=False,  # Hard-coded
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

        return train_loader, val_loader
    else:
        return train_loader, None
