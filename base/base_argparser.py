"""
Base class definition for all customized argument parsers.
Author: JiaWei Jiang
"""
import argparse
from abc import abstractmethod
from argparse import Namespace
from typing import Optional

__all__ = [
    "BaseArgParser",
]


class BaseArgParser:
    """Base class for all customized argument parsers."""

    def __init__(self) -> None:
        self.argparser = argparse.ArgumentParser()
        self._build()

    def parse(self) -> Namespace:
        """Return arguments driving the designated processes.

        Return:
            args: arguments driving the designated processes
        """
        args = self.argparser.parse_args()
        return args

    @abstractmethod
    def _build(self) -> None:
        """Build argument parser."""
        raise NotImplementedError

    def _str2bool(self, arg: str) -> Optional[bool]:
        """Convert boolean argument from string representation into
        bool.

        See https://stackoverflow.com/questions/15008758/.

        Parameters:
            arg: argument in string representation

        Return:
            True or False: argument in bool dtype
        """
        if isinstance(arg, bool):
            return arg
        if arg.lower() in ("true", "t", "yes", "y", "1"):
            return True
        elif arg.lower() in ("false", "f", "no", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Expect boolean representation.")
