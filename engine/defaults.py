"""
Boilerplate logic for controlling training and evaluation processes.
Author: JiaWei Jiang
"""
import argparse

from base.base_argparser import BaseArgParser

__all__ = [
    "TrainEvalArgParser",
]


class TrainEvalArgParser(BaseArgParser):
    """Argument parser for training and evaluation processes."""

    def __init__(self) -> None:
        super(TrainEvalArgParser, self).__init__()

    def _build(self) -> None:
        """Build argument parser."""
        self.argparser = argparse.ArgumentParser()
        self.argparser.add_argument(
            "--project-name",
            type=str,
            default=None,
            help="name of the project",
        )
        self.argparser.add_argument(
            "--exp-id",
            type=str,
            default=None,
            help="manully assigned experiment identifier",
        )
        self.argparser.add_argument(
            "--input-path",
            type=str,
            default=None,
            help="path of the input file",
        )
        self.argparser.add_argument(
            "--model-name",
            type=str,
            default=None,
            help="name of the model architecture",
        )
        self.argparser.add_argument(
            "--eval-on-test",
            type=self._str2bool,
            default=False,
            help="whether to evaluate on test set",
        )
        self.argparser.add_argument(
            "--use-wandb",
            type=self._str2bool,
            default=True,
            help="if True, training and eval processes are tracked with WandB",
        )
