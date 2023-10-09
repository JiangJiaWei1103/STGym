"""
Main script for training and evaluation processes.
Author: JiaWei Jiang

This script is used to run training and evaluation processes given the
specified CV scheme. Moreover, evaluation on unseen (test) data is
optional.

* [ ] Add wandb tracker via public method in `Experiment` (e.g.,
    `add_wnb_run` method).
* [ ] Feed priori graph structure when calling model `forward`.
* [ ] Derive `num_training_steps` elsewhere.
"""
import gc
import math
import warnings
from argparse import Namespace

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
        dp = DataProcessor(args.input_path, exp.dp_cfg["dataset_name"], **exp.dp_cfg["dp"])
        dp.run_before_splitting()
        data = dp.get_data_cv()
        priori_gs = dp.get_priori_gs()

        # Run CV
        cv = build_cv(**exp.dp_cfg["cv"])
        for fold, (tr_idx, val_idx) in enumerate(cv.split(X=data)):
            # Configure sub-entry for tracking current fold
            if args.use_wandb:
                tr_eval_run = exp.add_wnb_run(job_type="train_eval", name=f"fold{fold}")
            exp.log(f"\n== Train and Eval Process - Fold{fold} ==")

            # Build dataloaders
            data_tr, data_val = data[tr_idx, ...], data[val_idx, ...]
            data_tr, data_val, scaler = dp.run_after_splitting(data_tr, data_val)
            train_loader, val_loader = build_dataloaders(
                exp.dp_cfg["dataset_name"],
                data_tr,
                data_val,
                **exp.proc_cfg["dataloader"],
                **exp.dp_cfg["dataset"],
            )

            # Build model
            model_params = exp.model_params
            model = build_model(args.model_name, model_params)
            model.to(exp.proc_cfg["device"])
            if args.use_wandb:
                tr_eval_run.log({"model": {"n_params": count_params(model)}})
                tr_eval_run.watch(model, log="all", log_graph=True)

            # Build criterion
            loss_fn = build_criterion(**exp.proc_cfg["loss_fn"])

            # Build solvers
            optimizer = build_optimizer(model, **exp.proc_cfg["solver"]["optimizer"])
            num_training_steps = (
                math.ceil(len(train_loader) / exp.proc_cfg["dataloader"]["batch_size"]) * exp.proc_cfg["epochs"]
            )
            lr_skd = build_lr_scheduler(optimizer, num_training_steps, **exp.proc_cfg["solver"]["lr_skd"])

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
                "ckpt_path": exp.ckpt_path,
                "train_loader": train_loader,
                "eval_loader": val_loader,
                "priori_gs": priori_gs,
                "scaler": scaler,
                "use_wandb": args.use_wandb,
            }
            trainer = MainTrainer(**trainer_cfg)

            # Run main training and evaluation for one fold
            trainer.train_eval(fold)

            # Run evaluation on unseen test set
            if args.eval_on_test:
                data_test = dp.get_data_test()
                _, test_loader = build_dataloaders(
                    exp.dp_cfg["dataset_name"],
                    data_tr,
                    data_test,
                    **exp.proc_cfg["dataloader"],
                    **exp.dp_cfg["dataset"],
                )
                _ = trainer.test(fold, test_loader)

            trainer.profiler.summarize(log_wnb=True if args.use_wandb else False)

            # Dump output objects
            exp.dump_trafo(scaler, f"fold{fold}")

            # Free mem.
            del (
                data_tr,
                data_val,
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
                tr_eval_run.finish()


if __name__ == "__main__":
    # Parse arguments
    arg_parser = TrainEvalArgParser()
    args = arg_parser.parse()

    # Launch main function
    main(args)
