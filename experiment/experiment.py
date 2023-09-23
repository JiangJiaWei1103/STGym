"""
Experiment tracker.
Author: JiaWei Jiang

This file contains the definition of experiment tracker for experiment
configuration, message logging, object dumping, etc.
"""
from __future__ import annotations

import os
import pickle
from argparse import Namespace
from datetime import datetime
from types import TracebackType
from typing import Any, Dict, Optional, Type, Union

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.base import BaseEstimator
from torch.nn import Module

import wandb
from config.config import setup_dp, setup_model, setup_proc
from paths import DUMP_PATH
from utils.logger import Logger


class Experiment(object):
    """Experiment tracker.

    Parameters:
        args: arguments driving the designated process
        log_file: file to log experiment process
        infer: if True, the experiment is for inference only
    """

    cfg: Dict[str, Dict[str, Any]]
    model_params: Dict[str, Any]
    fit_params: Optional[Dict[str, Any]]
    exp_dump_path: str
    _cv_score: float = 0

    def __init__(
        self,
        args: Namespace,
        log_file: str = "train_eval.log",
        infer: bool = False,
    ) -> None:
        # Setup experiment identifier
        if args.exp_id is None:
            # Use time stamp as the experiment identifier
            args.exp_id = datetime.now().strftime("%m%d-%H_%M_%S")
        self.exp_id = args.exp_id

        self.args = args
        self.log_file = log_file
        self.infer = infer

        # Make buffer to dump output objects
        self._mkbuf()

        # Configure the experiment
        if not infer:
            self.dp_cfg = setup_dp()
            self.model_cfg = setup_model(args.model_name)
            self.proc_cfg = setup_proc()
            self._parse_model_cfg()
            self._agg_cfg()
        else:
            self._evoke_cfg()

        # Setup experiment logger
        if args.use_wandb:
            assert args.project_name is not None, "Please specify project name of wandb."
            self.exp_supr = wandb.init(
                project=args.project_name,
                config=self.cfg,
                group=self.exp_id,
                job_type="supervise",
                name="supr",
            )
        self.logger = Logger(logging_file=os.path.join(self.exp_dump_path, log_file)).get_logger()

    def __enter__(self) -> Experiment:
        self._log_exp_metadata()
        if self.args.use_wandb:
            self.exp_supr.finish()

        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_inst: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self._halt()

    def log(self, msg: str) -> None:
        """Log the provided message."""
        self.logger.info(msg)

    def dump_cfg(self, cfg: Dict[str, Any], file_name: str) -> None:
        """Dump configuration under corresponding path.

        Parameters:
            cfg: configuration
            file_name: config name with .yaml extension

        Return:
            None
        """
        file_name = file_name if file_name.endswith(".yaml") else f"{file_name}.yaml"
        dump_path = os.path.join(self.exp_dump_path, "config", file_name)
        with open(dump_path, "w") as f:
            yaml.dump(cfg, f)

    def dump_ndarr(self, arr: np.ndarray, file_name: str) -> None:
        """Dump np.ndarray to corresponding path.

        Parameters:
            arr: array to dump
            file_name: array name with .npy extension

        Return:
            None
        """
        dump_path = os.path.join(self.exp_dump_path, "preds", file_name)
        np.save(dump_path, arr)

    def dump_df(self, df: pd.DataFrame, file_name: str) -> None:
        """Dump DataFrame (e.g., feature imp) to corresponding path.

        Parameters:
            df: DataFrame to dump
            file_name: df name with .csv (by default) extension
        """
        if "." not in file_name:
            file_name = f"{file_name}.csv"
        dump_path = os.path.join(self.exp_dump_path, file_name)

        if file_name.endswith(".csv"):
            df.to_csv(dump_path, index=False)
        elif file_name.endswith(".parquet"):
            df.to_parquet(dump_path, index=False)

    def dump_model(self, model: Union[BaseEstimator, Module], file_name: str) -> None:
        """Dump the best model checkpoint to corresponding path.

        Parameters:
            model: well-trained estimator/model
            file_name: estimator/model name with .pkl/.pth extension

        Return:
            None
        """
        if isinstance(model, BaseEstimator):
            file_name = f"{file_name}.pkl"
        elif isinstance(model, Module):
            file_name = f"{file_name}.pth"
        dump_path = os.path.join(self.exp_dump_path, "models", file_name)

        if isinstance(model, BaseEstimator):
            with open(dump_path, "wb") as f:
                pickle.dump(model, f)
        elif isinstance(model, Module):
            torch.save(model.state_dict(), dump_path)

    def dump_trafo(self, trafo: Any, file_name: str) -> None:
        """Dump transfomer to corresponding path.

        Parameters:
            trafo: fitted transformer
            file_name: transformer name with .pkl extension

        Return:
            None
        """
        file_name = file_name if file_name.endswith(".pkl") else f"{file_name}.pkl"
        dump_path = os.path.join(self.exp_dump_path, "trafos", file_name)
        with open(dump_path, "wb") as f:
            pickle.dump(trafo, f)

    def set_cv_score(self, cv_score: float) -> None:
        """Set final CV score for recording.

        Parameters:
            cv_score: final CV score

        Return:
            None
        """
        self._cv_score = cv_score

    def _parse_model_cfg(self) -> None:
        """Configure model parameters and parameters passed to `fit`
        method if they're provided.

        Note that "fit_params" are always ignored for DL-based models.
        """
        self.model_params = self.model_cfg["model_params"]
        if self.model_cfg["fit_params"] is not None:
            self.fit_params = self.model_cfg["fit_params"]
        else:
            self.fit_params = None

    def _agg_cfg(self) -> None:
        """Aggregate separate configurations into the summarized one."""
        self.cfg = {
            "common": vars(self.args),
            "dp": self.dp_cfg,
            "model": self.model_params,
            "fit": self.fit_params,
            "proc": self.proc_cfg,
        }

    def _evoke_cfg(self) -> None:
        """Retrieve configuration of the pre-dumped experiment."""
        cfg_path = os.path.join(self.exp_dump_path, "config", "cfg.yaml")
        with open(cfg_path, "r") as f:
            self.cfg = yaml.full_load(f)
        self.dp_cfg = self.cfg["dp"]
        self.model_params = self.cfg["model"]
        self.fit_params = self.cfg["fit"]
        self.proc_cfg = self.cfg["proc"]

    def _mkbuf(self) -> None:
        """Make local buffer for experiment output dumping."""
        if not os.path.exists(DUMP_PATH):
            os.mkdir(DUMP_PATH)
        self.exp_dump_path = os.path.join(DUMP_PATH, self.exp_id)

        if self.infer:
            assert os.path.exists(self.exp_dump_path), "There exists no output objects for your specified experiment."
        else:
            if not os.path.exists(self.exp_dump_path):
                os.mkdir(self.exp_dump_path)

                # Create folders for output objects
                for sub_dir in ["config", "trafos", "models", "preds", "feats", "imps"]:
                    os.mkdir(os.path.join(self.exp_dump_path, sub_dir))
                for pred_type in ["oof", "holdout", "final"]:
                    os.mkdir(os.path.join(self.exp_dump_path, "preds", pred_type))
            else:
                print(f"{self.exp_dump_path} already exists!")

    def _run(self) -> None:
        pass

    def _log_exp_metadata(self) -> None:
        """Log metadata of the experiment to wandb."""
        self.log(f"=====Experiment {self.exp_id}=====")
        self.log(f"-> CFG: {self.cfg}\n")

    def _halt(self) -> None:
        if self.args.use_wandb:
            dump_entry = wandb.init(project=self.args.project_name, group=self.exp_id, job_type="dumping")

            # Log final CV score if exists
            if self._cv_score is not None:
                dump_entry.log({"cv_score": self._cv_score})

            # Push artifacts to remote
            artif = wandb.Artifact(name=self.args.model_name.upper(), type="output")
            artif.add_dir(self.exp_dump_path)
            dump_entry.log_artifact(artif)
            dump_entry.finish()

        self.log(f"=====End of Experiment {self.exp_id}=====")
