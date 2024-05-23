"""
Experiment tracker.
Author: JiaWei Jiang

This experiment tracker is mainly used to configure the experiment,
handle message logging, provide interface for output object dumping.
"""
from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Dict, Optional, Type, Union

import numpy as np
import pandas as pd
import rich
import torch
import wandb
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from rich.syntax import Syntax
from rich.tree import Tree
from sklearn.base import BaseEstimator
from torch.nn import Module
from wandb.sdk.lib import RunDisabled
from wandb.sdk.wandb_run import Run

from utils.common import dictconfig2dict
from utils.logger import Logger


class Experiment(object):
    """Experiment tracker.

    Args:
        cfg: The configuration driving the designated process.
        log_file: The file to log experiment process.
        infer: If True, the experiment is for inference only.

    Attributes:
        exp_id: The unique experiment identifier.
        exp_dump_path: Output path of the experiment.
        ckpt_path: Path of model checkpoints.
    """

    exp_dump_path: Path
    ckpt_path: Path
    _cv_score: float = 0

    def __init__(
        self,
        cfg: DictConfig,
        log_file: str = "train_eval.log",
        infer: bool = False,
    ) -> None:
        # Setup experiment identifier
        if cfg.exp_id is None:
            cfg.exp_id = datetime.now().strftime("%m%d-%H_%M_%S")
        self.exp_id = cfg.exp_id

        self.cfg = cfg
        self.log_file = log_file
        self.infer = infer

        # Make buffer to dump output objects
        self._mkbuf()

        # Configure the experiment
        if infer:
            self._evoke_cfg()
        else:
            self.data_cfg = cfg.data
            self.model_cfg = cfg.model
            self.trainer_cfg = cfg.trainer

        # Setup experiment logger
        if cfg.use_wandb:
            assert cfg.project_name is not None, "Please specify project name of wandb."
            self.exp_supr = self.add_wnb_run(cfg=cfg, job_type="supervise", name="supr")
        self.logger = Logger(logging_file=self.exp_dump_path / log_file).get_logger()

    def _mkbuf(self) -> None:
        """Make local buffer to dump experiment output objects."""
        # Create parent dump path
        dump_path = Path(self.cfg["paths"]["DUMP_PATH"])
        dump_path.mkdir(parents=True, exist_ok=True)

        self.exp_dump_path = dump_path / self.exp_id
        self.ckpt_path = self.exp_dump_path / "models"

        if self.infer:
            assert self.exp_dump_path.exists(), "There exists no output objects for your specified experiment."
        else:
            self.exp_dump_path.mkdir(parents=True, exist_ok=False)
            for sub_dir in ["config", "trafos", "models", "preds", "feats", "imps"]:
                sub_path = self.exp_dump_path / sub_dir
                sub_path.mkdir(parents=True, exist_ok=False)
            for pred_type in ["oof", "holdout", "final"]:
                sub_path = self.exp_dump_path / "preds" / pred_type
                sub_path.mkdir(parents=True, exist_ok=False)

    def _evoke_cfg(self) -> None:
        """Retrieve configuration of the pre-dumped experiment."""
        pass

    def __enter__(self) -> Experiment:
        self._log_cfg()
        if self.cfg.use_wandb:
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

    def dump_cfg(self, cfg: Union[DictConfig, Dict[str, Any]], file_name: str) -> None:
        """Dump configuration under corresponding path.

        Args:
            cfg: The configuration object.
            file_name: The config name with .yaml extension.
        """
        file_name = file_name if file_name.endswith(".yaml") else f"{file_name}.yaml"
        dump_path = self.exp_dump_path / "config" / file_name
        if isinstance(cfg, Dict):
            cfg = OmegaConf.create(cfg)
        OmegaConf.save(cfg, dump_path)

    def dump_ndarr(self, arr: np.ndarray, file_name: str) -> None:
        """Dump np.ndarray to corresponding path.

        Args:
            arr: The numpy array.
            file_name: The array name with .npy extension.
        """
        dump_path = self.exp_dump_path / "preds" / file_name
        np.save(dump_path, arr)

    def dump_df(self, df: pd.DataFrame, file_name: str) -> None:
        """Dump DataFrame (e.g., feature imp) to corresponding path.

        Args:
            df: The DataFrame.
            file_name: The df name with .csv (by default) extension.
        """
        if "." not in file_name:
            file_name = f"{file_name}.csv"
        dump_path = self.exp_dump_path / file_name

        if file_name.endswith(".csv"):
            df.to_csv(dump_path, index=False)
        elif file_name.endswith(".parquet"):
            df.to_parquet(dump_path, index=False)
        elif file_name.endswith(".pkl"):
            df.to_pickle(dump_path)

    def dump_model(self, model: Union[BaseEstimator, Module], file_name: str) -> None:
        """Dump the model checkpoint to corresponding path.

        Support dumping for torch model and most sklearn estimator
        instances.

        Args:
            model: The model checkpoint.
            file_name: The estimator/model name with .pkl/.pth extension.
        """
        if isinstance(model, BaseEstimator):
            file_name = f"{file_name}.pkl"
        elif isinstance(model, Module):
            file_name = f"{file_name}.pth"
        dump_path = self.exp_dump_path / "models" / file_name

        if isinstance(model, BaseEstimator):
            with open(dump_path, "wb") as f:
                pickle.dump(model, f)
        elif isinstance(model, Module):
            torch.save(model.state_dict(), dump_path)

    def dump_trafo(self, trafo: Any, file_name: str) -> None:
        """Dump data transfomer (e.g., scaler) to corresponding path.

        Args:
            trafo: The fitted data transformer.
            file_name: The transformer name with .pkl extension.
        """
        file_name = file_name if file_name.endswith(".pkl") else f"{file_name}.pkl"
        dump_path = self.exp_dump_path / "trafos" / file_name
        with open(dump_path, "wb") as f:
            pickle.dump(trafo, f)

    def set_cv_score(self, cv_score: float) -> None:
        """Set the final CV score for recording.

        Args:
            cv_score: The final CV score.
        """
        self._cv_score = cv_score

    def add_wnb_run(
        self,
        cfg: Optional[Union[DictConfig, Dict[str, Any]]] = None,
        job_type: Optional[str] = None,
        name: Optional[str] = None,
    ) -> Union[Run, RunDisabled, None]:
        """Initialize an wandb run for experiment tracking.

        Args:
            cfg: The experiment config. Note that the current random
                seed is recorded.
            job_type: The job type of run.
            name: The name of run.

        Returns:
            The wandb run to track the current experiment.
        """
        if cfg is not None and isinstance(cfg, DictConfig):
            cfg = dictconfig2dict(cfg)
        run = wandb.init(project=self.cfg.project_name, config=cfg, group=self.exp_id, job_type=job_type, name=name)

        return run

    def _log_cfg(self) -> None:
        """Log experiment config."""
        style = "dim"
        tree = Tree("CFG", style=style, guide_style=style)
        core_fields = ["project_name", "exp_id", "data", "model", "trainer"]
        aux_fields = ["n_seeds", "seed", "use_wandb", "one_fold_only", "paths"]
        for field in core_fields + aux_fields:
            branch = tree.add(field, style=style, guide_style=style)

            cfg_field = self.cfg[field]
            if isinstance(cfg_field, DictConfig):
                branch_content = OmegaConf.to_yaml(cfg_field, resolve=True)
            else:
                branch_content = str(cfg_field)

            branch.add(Syntax(branch_content, "yaml"))

        # Log config to stdout and log file
        self.log(f"===== Experiment {self.exp_id} =====")
        rich.print(tree)
        with open(self.exp_dump_path / self.log_file, "a") as f:
            rich.print(tree, file=f)

    def _halt(self) -> None:
        if self.cfg.use_wandb:
            dump_entry = self.add_wnb_run(None, job_type="dumping")

            # Log final CV score if exists
            if self._cv_score is not None:
                dump_entry.log({"cv_score": self._cv_score})

            # Push artifacts to remote (Deprecated)
            # artif = wandb.Artifact(name=self.model_name.upper(), type="output")
            # artif.add_dir(self.exp_dump_path)
            # dump_entry.log_artifact(artif)

            dump_entry.finish()

        self.log(f"===== End of Experiment {self.exp_id} =====")
