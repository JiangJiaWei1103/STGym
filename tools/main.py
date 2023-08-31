"""
Main script for training and evaluation processes.
Author: JiaWei Jiang

This script is used to run training and evaluation processes given the
specified CV scheme. Moreover, evaluation on unseen (test) data is
optional.
"""
import gc
import warnings
from argparse import Namespace

import pandas as pd

import wandb
from base.base_trainer import BaseTrainer
from criterion.build import build_criterion
from cv.build import build_cv
from data.build import build_dataloaders
from data.data_processor import DataProcessor
from engine.defaults import TrainEvalArgParser
from evaluating.build import build_evaluator
from experiment.experiment import Experiment
from modeling.build import build_model
from solver.build import build_lr_scheduler, build_optimizer
from trainer.trainer import MainTrainer
from utils.common import count_params
from utils.early_stopping import EarlyStopping

warnings.simplefilter("ignore")


def main(args: Namespace) -> None:
    """Run training and evaluation processes.

    Parameters:
        args: arguments driving training and evaluation processes

    Returns:
        None
    """
    # Configure experiment
    experiment = Experiment(args)

    with experiment as exp:
        exp.dump_cfg(exp.cfg, "cfg")

        # Prepare data
        dp = DataProcessor(
            args.input_path,
            **{**exp.dp_cfg["dp"], "dataset_name": exp.dp_cfg["dataset"]["name"]},
        )
        dp.run_before_splitting()
        df = dp.get_df()
        priori_gs = dp.get_priori_gs()

        # Run CV
        cv = build_cv(**exp.dp_cfg["cv"])
        for i, (tr_idx, val_idx) in enumerate(cv.split(X=df)):
            # Configure sub-entry for tracking current fold
            if args.use_wandb:
                sub_entry = wandb.init(
                    project=args.project_name,
                    group=exp.exp_id,
                    job_type="train_eval",
                    name=f"fold{i}",
                )
            exp.log(f"Training and evaluation process of fold{i} starts...")

            # Build dataloaders
            if isinstance(df, pd.DataFrame):
                df_tr, df_val = df.iloc[tr_idx, :], df.iloc[val_idx, :]
            else:
                df_tr, df_val = df[tr_idx, :], df[val_idx, :]
            df_tr, df_val, scaler = dp.run_after_splitting(df_tr, df_val)
            train_loader, val_loader = build_dataloaders(
                df_tr,
                df_val,
                **exp.proc_cfg["dataloader"],
                **exp.dp_cfg["dataset"],
            )

            # Build model
            model_params = exp.model_params
            if priori_gs is not None:
                model_params["priori_gs"] = priori_gs
            model = build_model(args.model_name, model_params)
            model.to(exp.proc_cfg["device"])
            if args.use_wandb:
                wandb.log({"model": {"n_params": count_params(model)}})
                wandb.watch(model, log="all", log_graph=True)

            # Build criterion
            loss_fn = build_criterion(**exp.proc_cfg["loss_fn"])

            # Build solvers
            optimizer = build_optimizer(model, **exp.proc_cfg)
            lr_skd = build_lr_scheduler(optimizer, **exp.proc_cfg)

            # Build early stopping tracker
            if exp.proc_cfg["es"]["patience"] != 0:
                es = EarlyStopping(exp.proc_cfg["es"]["patience"], exp.proc_cfg["es"]["mode"])
            else:
                es = None

            # Build evaluator
            evaluator = build_evaluator(**exp.proc_cfg["evaluator"])

            # Build trainer
            trainer: BaseTrainer = None
            trainer_cfg = {
                "logger": exp.logger,
                "proc_cfg": exp.proc_cfg,
                "model": model,
                "loss_fn": loss_fn,
                "optimizer": optimizer,
                "lr_skd": lr_skd,
                "es": es,
                "evaluator": evaluator,
                "train_loader": train_loader,
                "eval_loader": val_loader,
                "scaler": scaler,
                "use_wandb": args.use_wandb,
            }
            trainer = MainTrainer(**trainer_cfg)

            # Run main training and evaluation for one fold
            best_model, best_preds = trainer.train_eval(i)

            # Run evaluation on unseen test set
            if args.eval_on_test:
                df_test = dp.get_df_test()
                _, test_loader = build_dataloaders(
                    df_tr,
                    df_test,
                    **exp.proc_cfg["dataloader"],
                    **exp.dp_cfg["dataset"],
                )
                y_pred_test = trainer.test(i, test_loader)

            trainer.profiler.summarize(log_wnb=True if args.use_wandb else False)

            # Dump output objects
            exp.dump_trafo(scaler, f"fold{i}")
            exp.dump_model(best_model, f"fold{i}")

            # Free mem.
            del (
                df_tr,
                df_val,
                train_loader,
                val_loader,
                model,
                optimizer,
                lr_skd,
                es,
                evaluator,
                trainer,
            )
            _ = gc.collect()

            if args.use_wandb:
                sub_entry.finish()


if __name__ == "__main__":
    # Parse arguments
    arg_parser = TrainEvalArgParser()
    args = arg_parser.parse()

    # Launch main function
    main(args)
