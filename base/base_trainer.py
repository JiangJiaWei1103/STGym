"""
Base class definition for all customized trainers.
Author: JiaWei Jiang

* [ ] Add checkpoint tracker.
"""
import os
from abc import abstractmethod
from copy import deepcopy
from logging import Logger
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer, lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

import wandb
from evaluating.evaluator import Evaluator
from utils.common import Profiler
from utils.early_stopping import EarlyStopping


class BaseTrainer:
    """Base class for all customized trainers.

    Parameters:
        logger: message logger
        proc_cfg: hyperparameters for training and evaluation processes
        model: model instance
        loss_fn: loss criterion
        optimizer: optimization algorithm
        lr_skd: learning rate scheduler
        es: early stopping tracker
        evaluator: task-specific evaluator
        use_wandb: if True, training and evaluation processes are
            tracked with WandB
    """

    train_loader: DataLoader  # Temporary workaround
    profiler: Profiler = Profiler()

    def __init__(
        self,
        logger: Logger,
        proc_cfg: Dict[str, Any],
        model: Module,
        loss_fn: _Loss,
        optimizer: Optimizer,
        lr_skd: Union[_LRScheduler, lr_scheduler.ReduceLROnPlateau],
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
        self.evaluator = evaluator
        self.use_wandb = use_wandb

        self.device = proc_cfg["device"]
        self.epochs = proc_cfg["epochs"]
        self.ckpt_metric = proc_cfg["ckpt_metric"]

        self._iter = 0
        self._track_best_model = True

    def train_eval(self, proc_id: int) -> Tuple[nn.Module, Tensor]:
        """Run train and evaluation processes for either one fold or
        one random seed (commonly used when training on whole dataset).

        Parameters:
            proc_id: identifier of the current process, indicating
                current fold number or random seed.

        Return:
            None
        """
        val_loss_best = 1e18  # Monitored objective can be altered
        best_epoch = 0
        try:
            best_model = deepcopy(self.model)
        except RuntimeError as e:
            best_model = None
            self._track_best_model = False
            self.logger.warning("In-memoey best model tracking is disabled.")

        for epoch in range(self.epochs):
            train_loss = self._train_epoch()
            val_loss, val_result, _ = self._eval_epoch(datatype="val")

            # Adjust learning rate
            if self.lr_skd is not None:
                if isinstance(self.lr_skd, lr_scheduler.ReduceLROnPlateau):
                    self.lr_skd.step(val_loss)
                else:
                    self.lr_skd.step()

            # Track and log process result
            if isinstance(train_loss, dict):
                loss_msg = {}
                for loss_type, loss_val in train_loss.items():
                    if loss_type == "supr":
                        # Core supervision
                        loss_msg["train_loss"] = loss_val
                    else:
                        loss_msg[f"train_loss_{loss_type}"] = loss_val
            else:
                assert isinstance(train_loss, float)
                loss_msg = {"train_loss": train_loss}
            loss_msg["val_loss"] = val_loss
            if self.use_wandb:
                wandb.log(loss_msg)

            val_metric_msg = ""
            for metric, score in val_result.items():
                val_metric_msg += f"{metric.upper()} {round(score, 4)} | "
            self.logger.info(
                f"Epoch{epoch} | Training loss {loss_msg['train_loss']:.4f} | "
                f"Validation loss {val_loss:.4f} | {val_metric_msg}"
            )

            # Record the best checkpoint
            ckpt_metric_val = val_loss if self.ckpt_metric is None else val_result[self.ckpt_metric]
            if ckpt_metric_val < val_loss_best:
                self.logger.info(f"Validation performance improves at epoch {epoch}!!")
                val_loss_best = ckpt_metric_val
                if self._track_best_model:
                    best_model = deepcopy(self.model)
                else:
                    self._save_ckpt()
                best_epoch = epoch

            # Check early stopping is triggered or not
            if self.es is not None:
                # self.es.step(val_loss)
                self.es.step(ckpt_metric_val)
                if self.es.stop:
                    self.logger.info(f"Early stopping is triggered at epoch {epoch}, " f"training process is halted.")
                    break
        if self.use_wandb:
            wandb.log({"best_epoch": best_epoch})

        # Run final evaluation
        if not self._track_best_model:
            self._load_best_ckpt()
            best_model = self.model
        else:
            self.model = best_model
        final_prf_report, y_preds = self._eval_with_best()
        self._log_best_prf(final_prf_report)

        return best_model, y_preds

    def test(self, proc_id: int, test_loader: DataLoader) -> Tensor:
        """Run evaluation process on unseen test data.

        Parameters:
            proc_id: identifier of the current process, indicating
                current fold number.
            test_loader: test data loader

        Return:
            y_pred: prediction on test set
        """
        test_prf_report = {}

        self.eval_loader = test_loader
        eval_loss, eval_result, y_pred = self._eval_epoch(return_output=True, datatype="test", test=True)
        test_prf_report["test"] = eval_result
        self._log_best_prf(test_prf_report)

        return y_pred

    @abstractmethod
    def _train_epoch(self) -> Union[float, Dict[str, float]]:
        """Run training process for one epoch.

        Return:
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
        test: bool = False,
    ) -> Tuple[float, Dict[str, float], Optional[Tensor]]:
        """Run evaluation process for one epoch.

        Parameters:
            return_output: whether to return inference result of model
            datatype: type of the dataset to evaluate
            test: if evaluation is run on testing set, set it to True
                *Note: The setting is mainly used to disable DAE doping
                    during testing phase.

        Return:
            eval_loss_avg: average evaluation loss over batches
            eval_result: evaluation performance report
            y_pred: inference result
        """
        raise NotImplementedError

    def _eval_with_best(self) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Tensor]]:
        """Run final evaluation process with the best checkpoint.

        Return:
            final_prf_report: performance report of final evaluation
            y_preds: inference results on different datasets
        """
        final_prf_report = {}
        y_preds = {}

        self._disable_shuffle()
        val_loader = self.eval_loader

        for datatype, dataloader in {
            "train": self.train_loader,
            "val": val_loader,
        }.items():
            self.eval_loader = dataloader
            eval_loss, eval_result, y_pred = self._eval_epoch(return_output=True, datatype=datatype)
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

    def _save_ckpt(self, proc_id: Optional[int] = None, save_best_only: bool = True) -> None:
        """Save checkpoints.

        Parameters:
            proc_id: identifier of the current process, indicating
                current fold number or random seed
            save_best_only: only checkpoint of the best epoch is saved

        Return:
            None
        """
        if self.use_wandb:
            torch.save(
                self.model.state_dict(),
                os.path.join("./legacy/model_tmp.pt"),
            )
        else:
            torch.save(
                self.model.state_dict(),
                os.path.join("./legacy/model_tmp_local.pt"),
            )

    def _load_best_ckpt(self, proc_id: Optional[int] = None) -> None:
        """Load the best model checkpoint for final evaluation.

        The best checkpoint is loaded and assigned to `self.model`.

        Parameters:
            proc_id: identifier of the current process, indicating
                current fold number or random seed

        Return:
            None
        """
        model_name = self.model.name
        # model_cfg = setup_model(model_name)
        # Had better to handle the following addon in the child class
        # if self.priori_gs is not None:  # type: ignore
        # model_cfg = {**model_cfg, "priori_gs": self.priori_gs}  # type: ignore
        device = torch.device(self.device)
        # self.model = build_model(model_name, model_cfg)
        # self.model.load_state_dict(
        #     torch.load(
        #         os.path.join(self.dump_path, f"models/fold{proc_id}.pth"),
        #         map_location=device,
        #     )
        # )
        if self.use_wandb:
            self.model.load_state_dict(
                torch.load(
                    os.path.join("./legacy/model_tmp.pt"),
                    map_location=device,
                )
            )
        else:
            self.model.load_state_dict(
                torch.load(
                    os.path.join("./legacy/model_tmp_local.pt"),
                    map_location=device,
                )
            )
        self.model = self.model.to(device)

    def _log_best_prf(self, prf_report: Dict[str, Any]) -> None:
        """Log performance evaluated with the best model checkpoint.

        Parameters:
            prf_report: performance report

        Return:
            None
        """
        import json

        self.logger.info(">>>>> Performance Report - Best Ckpt <<<<<")
        self.logger.info(json.dumps(prf_report, indent=4))

        if self.use_wandb:
            wandb.log(prf_report)
