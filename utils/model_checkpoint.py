"""
Model checkpoint.
Author: JiaWei Jiang

The model checkpoint tracks only one quantity at a time.

* [ ] Support checkpointing on multiple metrics.
* [ ] Checkpoint on batch (i.e., iteration), not epoch.
* [ ] Checkpoint periodically.
"""
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Union

import torch
from torch.nn import Module


class ModelCheckpoint(object):
    """Model checkpooint.

    Parameters:
        ckpt_path: path to save model checkpoint
        ckpt_metric: quantity to monitor during training process
        ckpt_mode: determine the direction of metric improvement
        best_ckpt_mid: model identifier of the probably best checkpoint
            used to do the final evaluation
    """

    def __init__(self, ckpt_path: Union[Path, str], ckpt_metric: str, ckpt_mode: str, best_ckpt_mid: str) -> None:
        self.ckpt_path = ckpt_path
        self.ckpt_metric = ckpt_metric
        self.ckpt_mode = ckpt_mode
        self.best_ckpt_mid = best_ckpt_mid

        # Specify checkpoint direction
        self.ckpt_dir = -1 if ckpt_mode == "max" else 1

        # Initialize checkpoint status
        self.best_val_score = 1e18
        self.best_epoch = 0

    def step(
        self, epoch: int, model: Module, val_loss: float, val_result: Dict[str, float], last_epoch: bool = False
    ) -> None:
        """Update checkpoint status for the current epoch.

        Parameters:
            epoch: current epoch
            model: current model instance
            val_loss: validation loss
            val_result: evaluation result on validation set
            last_epoch: if True, current epoch is the last one
        """
        val_score = val_loss if self.ckpt_metric is None else val_result[self.ckpt_metric]
        val_score = val_score * self.ckpt_dir
        if val_score < self.best_val_score:  # type: ignore
            logging.info(f"Validation performance improves at epoch {epoch}!!")
            self.best_val_score = val_score
            self.best_epoch = epoch

            # Save model checkpoint
            mid = "loss" if self.ckpt_metric is None else self.ckpt_metric
            self._save_ckpt(model, mid)

        if last_epoch:
            self._save_ckpt(model, "last")

    def save_ckpt(self, model: Module, mid: Optional[str] = None) -> None:
        """Save the checkpoint.

        Parameters:
            model: current model instance
            mid: model identifer

        Return:
            None
        """
        self._save_ckpt(model, mid)

    def load_best_ckpt(self, model: Module, device: torch.device) -> Module:
        """Load and return the best model checkpoint for final evaluation.

        Parameters:
            model: current model instance
                *Note: Model weights are overrided by the best checkpoint
            device: device of the model instance

        Return:
            best_model: best model checkpoint
        """
        model = self._load_ckpt(model, device, self.best_ckpt_mid)

        return model

    def _save_ckpt(self, model: Module, mid: Optional[str] = None) -> None:
        """Save the model checkpoint.

        Parameters:
            model: current model instance
            mid: model identifer

        Return:
            None
        """
        model_file = "model.pth" if mid is None else f"model-{mid}.pth"
        torch.save(model.state_dict(), os.path.join(self.ckpt_path, model_file))

    def _load_ckpt(self, model: Module, device: torch.device, mid: str = "last") -> Module:
        """Load the model checkpoint.

        Parameters:
            model: current model instance
                *Note: Model weights are overrided by the best checkpoint
            device: device of the model instance
            mid: model identifier

        Return:
            model: model instance with the loaded weights
        """
        model_file = f"model-{mid}.pth"
        model.load_state_dict(torch.load(os.path.join(self.ckpt_path, model_file), map_location=device))

        return model
