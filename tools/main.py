"""
Main script for training and evaluation processes.
Author: JiaWei Jiang

This script is used to run training and evaluation processes given the
specified CV scheme.

* [ ] Dump prediction from the best ckpt or not.
* [ ] Write `cross_validate` function.
* [x] Use `instantiate` to build objects (e.g., model, optimizer).
"""
import gc
import math
import warnings

import hydra
import wandb
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig

from base.base_trainer import BaseTrainer
from config.config import get_seeds, seed_everything
from data.build import build_dataloader
from data.data_processor import DataProcessor
from experiment.experiment import Experiment
from trainer.trainer import MainTrainer
from utils.common import count_params
from utils.early_stopping import EarlyStopping

warnings.simplefilter("ignore")


def _cross_validate() -> None:
    pass


@hydra.main(config_path="../config", config_name="main")
def main(cfg: DictConfig) -> None:
    """Run training and evaluation processes.

    Args:
        cfg: The config driving training and evaluation processes.
    """
    # Configure experiment
    experiment = Experiment(cfg)

    with experiment as exp:
        exp.dump_cfg(exp.cfg, "main")

        # Prepare data
        dp = DataProcessor(**exp.data_cfg["dp"])
        dp.run_before_splitting()
        data = dp.get_data_cv()
        priori_gs = dp.get_priori_gs()
        aux_data = dp.get_aux_data()

        # Run cross-validation
        cv = instantiate(exp.data_cfg["cv"])
        one_fold_only = exp.cfg["one_fold_only"]
        seeds = get_seeds(exp.cfg["n_seeds"]) if exp.cfg["seed"] is None else [exp.cfg["seed"]]
        for s_i, seed in enumerate(seeds):
            exp.log(f"\nSeed the experiment with {seed}...")
            seed_everything(seed)
            cfg_seed = exp.cfg.copy()
            cfg_seed["seed"] = seed

            for fold, (tr_idx, val_idx) in enumerate(cv.split(X=data)):
                # Configure sub-entry for tracking current fold
                seed_name, fold_name = f"seed{s_i}", f"fold{fold}"
                proc_id = f"{seed_name}_{fold_name}"
                if exp.cfg["use_wandb"]:
                    tr_eval_run = exp.add_wnb_run(
                        cfg=cfg_seed,
                        job_type=fold_name if one_fold_only else seed_name,
                        name=seed_name if one_fold_only else fold_name,
                    )
                exp.log(f"== Train and Eval Process - Fold{fold} ==")

                # Build dataloaders
                data_tr, data_val = data[tr_idx], data[val_idx]
                data_tr, data_val, scaler = dp.run_after_splitting(data_tr, data_val)
                train_loader = build_dataloader(
                    data_tr, "train", exp.data_cfg["dataset"], **exp.trainer_cfg["dataloader"]
                )
                val_loader = build_dataloader(
                    data_val, "val", exp.data_cfg["dataset"], **exp.trainer_cfg["dataloader"]
                )

                # Build model
                model = instantiate(exp.model_cfg["model_params"])
                model.to(exp.trainer_cfg["device"])
                if exp.cfg["use_wandb"]:
                    wandb.log({"model": {"n_params": count_params(model)}})
                    wandb.watch(model, log="all", log_graph=True)

                # Build criterion
                loss_fn = instantiate(exp.trainer_cfg["loss_fn"])

                # Build solvers
                # Optimizer
                opt_partial = instantiate(exp.trainer_cfg["optimizer"])
                optimizer = opt_partial(params=model.parameters())
                # LR scheduler
                lr_skd_partial = instantiate(exp.trainer_cfg["lr_skd"])
                if lr_skd_partial is None:
                    lr_skd = None
                elif HydraConfig.get().runtime.choices["trainer/lr_skd"] == "cos":
                    num_training_steps = (
                        math.ceil(
                            len(train_loader.dataset)
                            / exp.trainer_cfg["dataloader"]["batch_size"]
                            / exp.trainer_cfg["grad_accum_steps"]
                        )
                        * exp.trainer_cfg["epochs"]
                    )
                    lr_skd = lr_skd_partial(optimizer=optimizer, T_max=num_training_steps)
                else:
                    lr_skd = lr_skd_partial(optimizer=optimizer)

                # Build early stopping tracker
                if exp.trainer_cfg["es"]["patience"] != 0:
                    es = EarlyStopping(exp.trainer_cfg["es"]["patience"], exp.trainer_cfg["es"]["mode"])
                else:
                    es = None

                # Build evaluator
                evaluator = instantiate(exp.trainer_cfg["evaluator"])

                # Build trainer
                trainer: BaseTrainer = None
                trainer = MainTrainer(
                    proc_cfg=exp.trainer_cfg,
                    logger=exp.logger,
                    model=model,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    lr_skd=lr_skd,
                    es=es,
                    ckpt_path=exp.ckpt_path,
                    evaluator=evaluator,
                    scaler=scaler,
                    train_loader=train_loader,
                    eval_loader=val_loader,
                    use_wandb=exp.cfg["use_wandb"],
                    priori_gs=priori_gs,
                    aux_data=aux_data,
                )

                # Run main training and evaluation for one fold
                trainer.train_eval(proc_id)

                # Run evaluation on unseen test set
                if exp.cfg["eval_on_test"]:
                    data_test = dp.get_data_test()
                    test_loader = build_dataloader(
                        data_test, "test", exp.data_cfg["dataset"], **exp.trainer_cfg["dataloader"]
                    )
                    _ = trainer.test(fold, test_loader)

                # Dump output objects
                if scaler is not None:
                    exp.dump_trafo(scaler, f"scaler_{proc_id}")
                for model_path in exp.ckpt_path.glob("*.pth"):
                    if "seed" in str(model_path) or "fold" in str(model_path):
                        continue

                    # Rename model file
                    model_file_name_dst = f"{model_path.stem}_{proc_id}.pth"
                    model_path_dst = exp.ckpt_path / model_file_name_dst
                    model_path.rename(model_path_dst)

                # Free mem.
                del (
                    data_tr,
                    data_val,
                    train_loader,
                    val_loader,
                    model,
                    opt_partial,
                    optimizer,
                    lr_skd_partial,
                    lr_skd,
                    evaluator,
                    trainer,
                )
                _ = gc.collect()

                if exp.cfg["use_wandb"]:
                    tr_eval_run.finish()
                if one_fold_only:
                    exp.log("Cross-validatoin stops at first fold!!!")
                    break


if __name__ == "__main__":
    # Launch main function
    main()
