"""
Customized local logger.
Author: JiaWei Jiang
"""
import logging
import sys
from typing import Optional


class Logger:
    """Customized logger.

    Parameters:
        logging_level: lowest-severity log message the logger handles
        logging_file: file stream for logging
            *Note: If `logging_file` isn't specified, message is only
                logged to system standard output.
    """

    _logger: logging.Logger = None

    def __init__(
        self,
        logging_level: str = "INFO",
        logging_file: Optional[str] = None,
    ):
        self.logging_level = logging_level
        self.logging_file = logging_file

        self._build_logger()

    def get_logger(self) -> logging.Logger:
        """Return customized logger."""
        return self._logger

    def _build_logger(self) -> None:
        """Build logger."""
        self._logger = logging.getLogger()
        self._logger.setLevel(self._get_level())
        self._add_handler()

    def _get_level(self) -> int:
        """Return lowest severity of the events the logger handles.

        Return:
            level: severity of the events
        """
        level = 0

        if self.logging_level == "DEBUG":
            level = logging.DEBUG
        elif self.logging_level == "INFO":
            level = logging.INFO
        elif self.logging_level == "WARNING":
            level = logging.WARNING
        elif self.logging_level == "ERROR":
            level = logging.ERROR
        elif self.logging_level == "CRITICAL":
            level = logging.CRITICAL

        return level

    def _add_handler(self) -> None:
        """Add stream and file (optional) handlers to logger."""
        s_handler = logging.StreamHandler(sys.stdout)
        self._logger.addHandler(s_handler)

        if self.logging_file is not None:
            f_handler = logging.FileHandler(self.logging_file, mode="a")
            self._logger.addHandler(f_handler)
