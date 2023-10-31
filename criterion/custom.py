"""
Custom loss criterion definitions.
Author: JiaWei Jiang

If users want to use customized loss criterion, the corresponding class
should be defined in this file.
"""
import numpy as np
import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss


class MaskedLoss(_Loss):
    """Base loss criterion with masking mechanism.

    Parameters:
        name: name of the base loss criterion
        masked_val: value to mask when deriving loss
    """

    def __init__(self, name: str = "l1", masked_val: float = 0.0):
        super(MaskedLoss, self).__init__()

        self.name = name
        self.masked_val = masked_val

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        mask = self._get_mask(y_true)
        base_loss = self._get_base_loss(y_true, y_pred)

        loss = base_loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        loss = torch.mean(loss)

        return loss

    def _get_mask(self, y_true: Tensor) -> Tensor:
        """Generate and return mask."""
        if np.isnan(self.masked_val):
            mask = ~torch.isnan(y_true)
        else:
            eps = 5e-5
            mask = ~torch.isclose(y_true, torch.tensor(self.masked_val).expand_as(y_true).to(y_true.device), atol=eps, rtol=0.)
            #mask = y_true != self.masked_val

        mask = mask.float()
        mask /= torch.mean(mask)  # Adjust non-masked entry weights
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

        return mask

    def _get_base_loss(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        """Derive and return loss using base loss criterion."""
        base_loss: Tensor = None
        if self.name == "l1":
            base_loss = torch.abs(y_pred - y_true)
        elif self.name == "l2":
            base_loss = torch.square(y_pred - y_true)
        elif self.name == "mape":
            base_loss = torch.abs(y_pred - y_true) / y_true
        elif self.name == "huber":
            base_loss = torch.nn.HuberLoss()(y_pred, y_true)

        return base_loss
