"""
Dataloader building logic.
Author: JiaWei Jiang

This file contains the basic logic of building dataloaders for training
and evaluation processes.

* [ ] Modify dataloader building logic (maybe one at a time).
"""
from typing import Any

import numpy as np
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig
from torch.utils.data import DataLoader


def build_dataloader(data: np.ndarray, data_split: str, dataset_cfg: DictConfig, **dataloader_cfg: Any) -> DataLoader:
    """Build and return dataloader.

    Args:
        data: Data to be fed into torch Dataset.
        data_split: The data split.
        dataset_cfg: The hyperparameters of dataset.
        dataloader_cfg: The hyperparameters of dataloader.

    Returns:
        A dataloader of the given data split.
    """
    dataset = instantiate(dataset_cfg)
    collate = None
    shuffle = dataloader_cfg["shuffle"] if data_split == "train" else False
    dataloader = DataLoader(
        dataset(data),
        batch_size=dataloader_cfg["batch_size"],
        shuffle=shuffle,
        num_workers=dataloader_cfg["num_workers"],
        collate_fn=collate,
        pin_memory=dataloader_cfg["pin_memory"],
        drop_last=dataloader_cfg["drop_last"],
    )

    return dataloader
