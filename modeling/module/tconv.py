"""
Temporal convolution modules.
Author: JiaWei Jiang
"""
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TConvBaseModule(nn.Module):
    """Temporal convolution base module.

    Following utilities are supported:
    1. Calculate the receptive field of the network.
    2. Pad the input sequence to the receptive field along the
        temporal dimension.
    """

    def __init__(self) -> None:
        super(TConvBaseModule, self).__init__()

    @property
    def receptive_field(self) -> int:
        return self._receptive_field

    def _set_receptive_field(self, n_layers: int, dilation_factor: int, kernel_size: int) -> None:
        """Calculate and set the receptive field.

        Parameters:
            n_layers: number of stacked layers
            dilation_factor: dilation exponential base
            kernel_size: kernel size
        """
        if dilation_factor > 1:
            self._receptive_field = int(
                1 + (kernel_size - 1) * (dilation_factor**n_layers - 1) / (dilation_factor - 1)
            )
        else:
            self._receptive_field = n_layers * (kernel_size - 1) + 1

    def _pad_seq_to_rf(self, x: Tensor) -> Tensor:
        """Pad the input sequence to the receptive field along the
        temporal dimension.

        Parameters:
            x: input sequence

        Return:
            x_pad: padded sequence

        Shape:
            x: (*, L), where * denotes any dimension and L denotes the
                input sequence length
            x_pad: (*, self.receptive_field)
        """
        in_len = x.shape[-1]
        x_pad = F.pad(x, (self.receptive_field - in_len, 0))

        return x_pad
