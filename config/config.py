"""
Experiment configuration logic.
Author: JiaWei Jiang

Utility functions for setting up experiments, including generating a
unique experiment identifier, seeding the experiment, etc.
"""
import os
import random
import string
from typing import List

import numpy as np
import torch


def gen_exp_id(model_name: str) -> str:
    """Generate an unique experiment identifier.

    Args:
        model_name: The name of model architecture.

    Returns:
        An unique experiment identifier.
    """
    chars = string.ascii_lowercase + string.digits
    exp_id = "".join(random.SystemRandom().choice(chars) for _ in range(8))
    exp_id = f"{model_name}-{exp_id}"

    return exp_id


def get_seeds(n_seeds: int = 3) -> List[int]:
    """Generate and return a list of random seeds.

    Args:
        n_seeds: The number of seeds.

    Returns:
        A list of random seeds.
    """
    seeds = [random.randint(1, 2**32 - 1) for _ in range(n_seeds)]

    return seeds


def seed_everything(seed: int) -> None:
    """Seed the current experiment.

    Note that it can't always guarantee reproducibility.

    Args:
        seed: The automatically generated or manually specified seed.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # When running with cudnn backend
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
