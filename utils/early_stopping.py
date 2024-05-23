"""
Early _stopping tracker.
Author: JiaWei Jiang

This file contains the definition of early _stopping to prevent overfit
or boost modeling efficiency.
"""
from typing import Optional


class EarlyStopping(object):
    """Monitor whether the specified metric improves or not. If metric
    doesn't improve for the `patience` epochs, then the training and
    evaluation processes will _stop early.

    *Note: Applying ES might have a risk of overfitting on validation
    set.

    Args:
        patience: tolerance for number of epochs when model can't
                  improve the specified score (e.g., loss, metric)
        mode: performance determination mode, the choices can be:
            {'min', 'max'}
        tr_loss_thres: _stop training immediately once training loss
            reaches this threshold
    """

    _best_score: float
    _stop: bool
    _wait_count: int

    def __init__(
        self,
        patience: int = 10,
        mode: str = "min",
        tr_loss_thres: Optional[float] = None,
    ):
        self.patience = patience
        self.mode = mode
        self.tr_loss_thres = tr_loss_thres
        self._setup()

    def step(self, score: float) -> None:
        """Update states of es tracker.

        Args:
            score: specified score in the current epoch

        Returns:
            None
        """
        if self.tr_loss_thres is not None:
            if score <= self.tr_loss_thres:
                self._stop = True
        else:
            score_adj = score if self.mode == "min" else -score
            if score_adj < self._best_score:
                self._best_score = score_adj
                self._wait_count = 0
            else:
                self._wait_count += 1

            if self._wait_count >= self.patience:
                self._stop = True

    @property
    def stop(self) -> bool:
        """If True, early stopping is triggered."""
        return self._stop

    def _setup(self) -> None:
        """Setup es tracker."""
        if self.mode == "min":
            self._best_score = 1e18
        elif self.mode == "max":
            self._best_score = -1 * 1e-18
        self._stop = False
        self._wait_count = 0
