"""
Evaluator building logic.
Author: JiaWei Jiang

This file contains the basic logic of building evaluator for evaluation
process.
"""
from typing import List, Optional

from .evaluator import Evaluator


def build_evaluator(
    eval_metrics: List[str],
    horiz_cuts: Optional[List[int]] = None,
) -> Evaluator:
    """Build and return the evaluator.

    Parameters:
        metric_names: evaluation metrics
        horiz_cuts: predicting horizon cutoff
            *Note: This argument indicates how many predicting
                horizons are considered when deriving performance,
                and it's only used in multi-horizon scenario.
    Return:
        evaluator: evaluator
    """
    evaluator = Evaluator(eval_metrics, horiz_cuts)

    return evaluator
