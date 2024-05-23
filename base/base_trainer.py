"""
Base class definition for all customized trainers.
Author: JiaWei Jiang

* [ ] Design better profiling workflow.
* [ ] Add checkpoint tracker.
"""
from abc import abstractmethod
from logging import Logger
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import wandb
from torch import Tensor
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer, lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from evaluating.evaluator import Evaluator
from utils.common import Profiler
from utils.early_stopping import EarlyStopping
from utils.model_checkpoint import ModelCheckpoint


class BaseTrainer:
    """Base class for all customized trainers.

    Args:
        logger: message logger
        proc_cfg: hyperparameters for training and evaluation processes
        model: model instance
        loss_fn: loss criterion
        optimizer: optimization algorithm
        lr_skd: learning rate scheduler
        ckpt_path: path to save model checkpoints
        es: early stopping tracker
        evaluator: task-specific evaluator
        use_wandb: if True, training and evaluation processes are
            tracked with wandb
    """

    train_loader: DataLoader  # Tmp. workaround
    eval_loader: DataLoader  # Tmp. workaround
    profiler: Profiler = Profiler()

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
        use_wandb: bool,
    ):
        self.logger = logger
        self.proc_cfg = proc_cfg
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_skd = lr_skd
        self.es = es
        self.ckpt_path = ckpt_path
        self.evaluator = evaluator
        self.use_wandb = use_wandb

        self.device = proc_cfg["device"]
        self.epochs = proc_cfg["epochs"]
        # If True, learning rate scheduler steps per batches
        # self.step_per_batch = proc_cfg["solver"]["lr_skd"]["step_per_batch"]
        self.step_per_batch = True

        # Model checkpoint
        self.model_ckpt = ModelCheckpoint(ckpt_path, **proc_cfg["model_ckpt"])

        self._iter = 0
        self._track_best_model = True  # (Deprecated)

    def train_eval(self, proc_id: str) -> Tuple[Module, Tensor]:
        """Run train and evaluation processes for either one fold or
        one random seed (commonly used when training on whole dataset).

        Args:
            proc_id: identifier of the current process, indicating
                current fold number or random seed.

        Returns:
            None
        """
        best_model, best_y_pred = None, None
        for epoch in range(self.epochs):
            train_loss = self._train_epoch()
            val_loss, val_result, _ = self._eval_epoch(datatype="val")

            # Adjust learning rate
            if not self.step_per_batch and self.lr_skd is not None:
                if isinstance(self.lr_skd, lr_scheduler.ReduceLROnPlateau):
                    self.lr_skd.step(val_loss)
                else:
                    self.lr_skd.step()

            # Track and log process result (by epoch)
            self._log_proc(epoch, train_loss, val_loss, val_result)

            # Record the best checkpoint
            self.model_ckpt.step(
                epoch, self.model, val_loss, val_result, last_epoch=False if epoch != self.epochs - 1 else True
            )

            # Check early stopping is triggered or not
            if self.es is not None:
                self.es.step(val_loss)
                if self.es.stop:
                    self.logger.info(f"Early stopping is triggered at epoch {epoch}, " f"training process is halted.")
                    break
        if self.use_wandb:
            wandb.log({"best_epoch": self.model_ckpt.best_epoch + 1})

        # Run final evaluation
        final_prf_report, y_preds = self._run_final_eval()
        self._log_best_prf(final_prf_report)

        return best_model, y_preds

    def test(self, proc_id: int, test_loader: DataLoader) -> Tensor:
        """Run evaluation process on unseen test data.

        Args:
            proc_id: identifier of the current process, indicating
                current fold number
            test_loader: test data loader

        Returns:
            y_pred: prediction on test set
        """
        self.eval_loader = test_loader
        _, eval_result, y_pred = self._eval_epoch(return_output=True, datatype="test")
        test_prf_report = {"test": eval_result}
        self._log_best_prf(test_prf_report)

        return y_pred

    @abstractmethod
    def _train_epoch(self) -> Union[float, Dict[str, float]]:
        """Run training process for one epoch.

        Returns:
            train_loss_avg: average training loss over batches
                *Note: If multitask is used, returned object will be
                    a dictionary containing losses of subtasks and the
                    total loss.
        """
        raise NotImplementedError

    @abstractmethod
    def _eval_epoch(
        self,
        return_output: bool = False,
        datatype: str = "val",
    ) -> Tuple[float, Dict[str, float], Optional[Tensor]]:
        """Run evaluation process for one epoch.

        Args:
            return_output: whether to return inference result of model
            datatype: type of the dataset to evaluate

        Returns:
            eval_loss_avg: average evaluation loss over batches
            eval_result: evaluation performance report
            y_pred: inference result
        """
        raise NotImplementedError

    def _log_proc(
        self,
        epoch: int,
        train_loss: Union[float, Dict[str, float]],
        val_loss: float,
        val_result: Dict[str, float],
        proc_id: Optional[str] = None,
    ) -> None:
        """Log message of training process.

        Args:
            epoch: current epoch number
            train_loss: training loss
            val_loss: validation loss
            val_result: evaluation performance report
            proc_id: identifier of the current process, indicating
                current fold number, random seed, or dataset id.
        """
        # Construct training loss message
        train_loss_msg = ""
        if isinstance(train_loss, float):
            train_loss_msg = f"Training loss {train_loss:.4f}"
        else:
            for loss_k, loss_v in train_loss.items():
                loss_name = loss_k.split("_")[0].capitalize()
                train_loss_msg += f"{loss_name} loss {round(loss_v, 4)} | "

        # Construct eval prf message
        val_metric_msg = ""
        for metric, score in val_result.items():
            val_metric_msg += f"{metric.upper()} {round(score, 4)} | "

        self.logger.info(f"Epoch{epoch} | {train_loss_msg} | " f"Validation loss {val_loss:.4f} | {val_metric_msg}")
        if self.use_wandb:
            # Process loss dict and log
            log_dict = train_loss if isinstance(train_loss, dict) else {"train_loss": train_loss}
            log_dict["val_loss"] = val_loss

            # ===
            # Add metric tracking
            # ===

            if proc_id is not None:
                log_dict = {f"{k}_{proc_id}": v for k, v in log_dict.items()}

            wandb.log(log_dict)

    def _run_final_eval(self) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Tensor]]:
        """Run final evaluation process with the best checkpoint.

        Returns:
            final_prf_report: performance report of final evaluation
            y_preds: inference results on different datasets
        """
        # Load the best model checkpoint
        self.model = self.model_ckpt.load_best_ckpt(self.model, self.device)

        # Reconstruct dataloaders
        self._disable_shuffle()
        val_loader = self.eval_loader

        final_prf_report, y_preds = {}, {}
        for datatype, dataloader in {
            "train": self.train_loader,
            "val": val_loader,
        }.items():
            self.eval_loader = dataloader
            _, eval_result, y_pred = self._eval_epoch(return_output=True, datatype=datatype)
            final_prf_report[datatype] = eval_result
            y_preds[datatype] = y_pred

        return final_prf_report, y_preds

    def _disable_shuffle(self) -> None:
        """Disable shuffle in train dataloader for final evaluation."""
        self.train_loader = DataLoader(
            self.train_loader.dataset,
            batch_size=self.train_loader.batch_size,
            shuffle=False,  # Reset shuffle to False
            num_workers=self.train_loader.num_workers,
            collate_fn=self.train_loader.collate_fn,
        )

    def _log_best_prf(self, prf_report: Dict[str, Any]) -> None:
        """Log performance evaluated with the best model checkpoint.

        Args:
            prf_report: performance report

        Returns:
            None
        """
        import json

        self.logger.info(">>>>> Performance Report - Best Ckpt <<<<<")
        self.logger.info(json.dumps(prf_report, indent=4))

        if self.use_wandb:
            wandb.log(prf_report)
