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
from .Final_STGCN import STGCN
from .Final_GWNet import GWNet
from .Final_ASTGCN import ASTGCN
from .Final_STSGCN import STSGCN
from .Final_AGCRN import AGCRN
from .Final_GMAN import GMAN
from .Final_MTGNN import MTGNN
from .Final_DGCRN import DGCRN
from .Final_GTS import GTS
from .Final_STNorm import STNorm
from .Final_STID import STID
from .Final_LST_Skip import LST_Skip


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
    elif model_name == "STGCN":
        model = STGCN(**model_cfg)
    elif model_name == "GWNet":
        model = GWNet(**model_cfg)
    elif model_name == "ASTGCN":
        model = ASTGCN(**model_cfg)
    elif model_name == "STSGCN":
        model = STSGCN(**model_cfg)
    elif model_name == "AGCRN":
        model = AGCRN(**model_cfg)
    elif model_name == "GMAN":
        model = GMAN(**model_cfg)  
    elif model_name == "MTGNN":
        model = MTGNN(**model_cfg)
    elif model_name == "DGCRN":
        model = DGCRN(**model_cfg)
    elif model_name == "GTS":
        model = GTS(**model_cfg)
    elif model_name == "STNorm":
        model = STNorm(**model_cfg)
    elif model_name == "STID":
        model = STID(**model_cfg)
    elif model_name == "LST_Skip":
        model = LST_Skip(**model_cfg)
    else:
        raise RuntimeError(f"{model_name} isn't registered.")

    return model