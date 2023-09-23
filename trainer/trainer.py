"""
Custom trainer definitions for different training processes.
Author: JiaWei Jiang

This file contains diversified trainers, whose training logics are
inherited from `BaseTrainer`.

* [ ] Pack input data in Dict.
* [ ] Fuse grad clipping mechanism into solver.
"""
import gc
from logging import Logger
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer, lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from base.base_trainer import BaseTrainer
from evaluating.evaluator import Evaluator
from utils.early_stopping import EarlyStopping
from utils.scaler import MaxScaler, StandardScaler


class MainTrainer(BaseTrainer):
    """Main trainer.

    It's better to define different trainers for different models if
    there's a significant difference within training and evaluation
    processes (e.g., model input, advanced data processing, graph node
    sampling, customized multitask criterion definition).

    Parameters:
        logger: message logger
        proc_cfg: hyperparameters for training and evaluation processes
        model: model instance
        loss_fn: loss criterion
        optimizer: optimization algorithm
        lr_scheduler: learning rate scheduler
        es: early stopping tracker
        scaler: scaling object
        train_loader: training data loader
        eval_loader: validation data loader
        use_wandb: if True, training and evaluation processes are
            tracked with WandB
    """

    def __init__(
        self,
        logger: Logger,
        proc_cfg: Dict[str, Any],
        model: Module,
        loss_fn: _Loss,
        optimizer: Optimizer,
        lr_skd: Union[_LRScheduler, lr_scheduler.ReduceLROnPlateau],
        ckpt_path: Union[Path, str],
        es: EarlyStopping,
        evaluator: Evaluator,
        scaler: Union[MaxScaler, StandardScaler],
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
        use_wandb: bool = True,
    ):
        super(MainTrainer, self).__init__(
            logger,
            proc_cfg,
            model,
            loss_fn,
            optimizer,
            lr_skd,
            ckpt_path,
            es,
            evaluator,
            use_wandb,
        )
        self.train_loader = train_loader
        self.eval_loader = eval_loader if eval_loader else train_loader
        self.scaler = scaler
        self.rescale = proc_cfg["loss_fn"]["rescale"]

        # Curriculum learning
        if self.proc_cfg["loss_fn"]["cl"] is not None:
            self._cl = CLTracker(**self.proc_cfg["loss_fn"]["cl"])
        else:
            self._cl = None

    def _train_epoch(self) -> float:
        """Run training process for one epoch.

        Return:
            train_loss_avg: average training loss over batches
        """
        train_loss_total = 0

        self.model.train()
        self.profiler.start("train")
        for i, batch_data in enumerate(tqdm(self.train_loader)):
            self.optimizer.zero_grad(set_to_none=True)

            # Retrieve batched raw data
            x = batch_data["X"].to(self.device)
            y = batch_data["y"].to(self.device)
            tid = None if "tid" not in batch_data else batch_data["tid"].to(self.device)
            diw = None if "diw" not in batch_data else batch_data["diw"].to(self.device)

            # Forward pass and derive loss
            output, *_ = self.model(
                x,
                tid=tid,
                diw=diw,
                ycl=y,
                batches_seen=self._iter,
                task_level=None if self._cl is None else self._cl.get_task_lv(),
            )

            # Inverse transform to the original scale
            if self.rescale:
                output = self.scaler.inverse_transform(output)
                y = self.scaler.inverse_transform(y)

            # Derive loss
            if y.dim() == 4:
                y = y[..., 0]
            if self._cl is not None:
                task_lv = self._cl.get_task_lv()
                loss = self.loss_fn(output[:, :task_lv, :], y[:, :task_lv, :])
                self._cl.step()
            else:
                loss = self.loss_fn(output, y)

            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)

            self.optimizer.step()
            self._iter += 1
            train_loss_total += loss.item()

            # Free mem.
            del x, y, output
            _ = gc.collect()

        self.profiler.stop()
        train_loss_avg = train_loss_total / len(self.train_loader)

        return train_loss_avg

    @torch.no_grad()
    def _eval_epoch(
        self,
        return_output: bool = False,
        datatype: str = "val",
    ) -> Tuple[float, Dict[str, float], Optional[Tensor]]:
        """Run evaluation process for one epoch.

        Parameters:
            return_output: whether to return inference result of model
            datatype: type of the dataset to evaluate

        Return:
            eval_loss_avg: average evaluation loss over batches
            eval_result: evaluation performance report
            y_pred: inference result
        """
        eval_loss_total = 0
        y_true = None
        y_pred = None

        self.model.eval()
        self.profiler.start(proc_type=datatype)
        for i, batch_data in enumerate(self.eval_loader):
            # Retrieve batched raw data
            x = batch_data["X"].to(self.device)
            y = batch_data["y"].to(self.device)
            tid = None if "tid" not in batch_data else batch_data["tid"].to(self.device)
            diw = None if "diw" not in batch_data else batch_data["diw"].to(self.device)

            # Forward pass
            output, *_ = self.model(x, tid=tid, diw=diw)

            # Derive loss
            if y.dim() == 4:
                y = y[..., 0]
            if self.rescale:
                loss = self.loss_fn(
                    self.scaler.inverse_transform(output),
                    self.scaler.inverse_transform(y),
                )
            else:
                loss = self.loss_fn(output, y)
            eval_loss_total += loss.item()

            # Record batched output
            if i == 0:
                y_true = torch.squeeze(y).detach().cpu()
                y_pred = torch.squeeze(output).detach().cpu()
            else:
                # Avoid situation that batch_size is just equal to 1
                y_true = torch.cat((y_true, y.detach().cpu()))
                y_pred = torch.cat((y_pred, output.detach().cpu()))

            del x, y, output
            _ = gc.collect()

        self.profiler.stop(record=True if datatype != "train" else False)
        eval_loss_avg = eval_loss_total / len(self.eval_loader)

        # Run evaluation with the specified evaluation metrics
        eval_result = self.evaluator.evaluate(y_true, y_pred, self.scaler)

        if return_output:
            return eval_loss_avg, eval_result, y_pred
        else:
            return eval_loss_avg, eval_result, None


class CLTracker(object):
    """Tracker for curriculum learning.

    Parameters:
        lv_up_period: task levels up every `lv_up_period` iterations if
            the current task level is lower than `task_lv_max`
        task_lv_max: hardest task level
    """

    def __init__(self, lv_up_period: int = 2500, task_lv_max: int = 12):
        self.lv_up_period = lv_up_period
        self.task_lv_max = task_lv_max

        self._iter = 0
        self._task_lv = 1

    def get_iter(self) -> int:
        """Return the current iteration."""
        return self._iter

    def get_task_lv(self) -> int:
        """Return the current task level.

        Return:
            self._task_lv: current task level
        """
        return self._task_lv

    def step(self) -> None:
        """Step forward once current iteration ends."""
        self._iter += 1

        if self._task_lv < self.task_lv_max and self._iter % self.lv_up_period == 0:
            self._task_lv += 1
