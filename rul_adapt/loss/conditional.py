from typing import Any, List, Tuple

import torch
import torchmetrics
from torch import nn


class ConditionalAdaptionLoss(torchmetrics.Metric):
    is_differentiable = True
    higher_is_better = False
    full_state_update = True

    loss: torch.Tensor

    def __init__(
        self,
        adaption_losses: List[torchmetrics.Metric],
        fuzzy_sets: List[Tuple[float, float]],
        mean_over_sets: bool = True,
    ) -> None:
        super().__init__()

        self.adaption_losses = nn.ModuleList(adaption_losses)  # registers parameters
        self.fuzzy_sets = fuzzy_sets
        self.mean_over_sets = mean_over_sets

        self.add_state("loss", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        source: torch.Tensor,
        source_preds: torch.Tensor,
        target: torch.Tensor,
        target_preds: torch.Tensor,
    ) -> None:
        for fuzzy_set, adaption_loss in zip(self.fuzzy_sets, self.adaption_losses):
            selected_source = source[_membership(source_preds, fuzzy_set)]
            selected_target = target[_membership(target_preds, fuzzy_set)]
            self.loss = self.loss + adaption_loss(selected_source, selected_target)

    def compute(self) -> torch.Tensor:
        if self.mean_over_sets:
            return self.loss / len(self.adaption_losses)
        else:
            return self.loss


def _membership(preds: torch.Tensor, fuzzy_set: Tuple[float, float]) -> torch.Tensor:
    preds = preds.squeeze() if len(preds.shape) > 1 else preds
    membership = (preds >= fuzzy_set[0]) & (preds < fuzzy_set[1])

    return membership
