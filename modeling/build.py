"""
Model architecture building logic.
Author: JiaWei Jiang

The file contains a single function for model name switching and model
architecture building based on the model configuration.

To add in new model, users need to design custom model architecture,
put the file under the same directory, and import the corresponding
model below (i.e., import the separately pre-defined model arch.).
"""
from typing import Any, Dict

from torch.nn import Module

from .Final import HARDPurG
from .sotas.DCRNN import DCRNN
from .sotas.GWNet import GWNet
from .sotas.MTGNN import MTGNN


def build_model(model_name: str, model_cfg: Dict[str, Any]) -> Module:
    """Build and return the specified model architecture.

    Parameters:
        model_name: name of model architecture
        model_cfg: hyperparameters of the specified model

    Return:
        model: model instance
    """
    model: Module
    if model_name == "HARDPurG":
        model = HARDPurG(**model_cfg)
    elif model_name == "DCRNN":
        model = DCRNN(**model_cfg)
    elif model_name == "GWNet":
        model = GWNet(**model_cfg)
    elif model_name == "MTGNN":
        model = MTGNN(**model_cfg)
    else:
        raise RuntimeError(f"{model_name} isn't registered.")

    return model
