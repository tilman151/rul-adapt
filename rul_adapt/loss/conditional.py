"""A module for conditional unsupervised domain adaption losses."""

from typing import List, Tuple, Sequence

import torch
import torchmetrics
from torch import nn


class ConditionalAdaptionLoss(torchmetrics.Metric):
    """The Conditional Adaptions loss is a combination of multiple losses, each of
    which is only applied to a subset of the incoming data.

    The subsets are defined by fuzzy sets with a rectangular membership function. The
    prediction for each sample is checked against the fuzzy sets, and the
    corresponding loss is applied to the sample. The combined loss can be set as the
    sum of all components or their mean."""

    is_differentiable = True
    higher_is_better = False
    full_state_update = True

    loss: torch.Tensor
    batch_counter: torch.Tensor

    def __init__(
        self,
        adaption_losses: Sequence[torchmetrics.Metric],
        fuzzy_sets: List[Tuple[float, float]],
        mean_over_sets: bool = True,
    ) -> None:
        """
        Create a new Conditional Adaption loss over fuzzy sets.

        The fuzzy sets' boundaries are inclusive on the left and exclusive on the right.
        The left boundary is supposed to be smaller than the right boundary.

        This loss should not be used as a way to accumulate its value over multiple
        batches.

        Args:
            adaption_losses: The list of losses to be applied to the subsets.
            fuzzy_sets: The fuzzy sets to be used for subset membership.
            mean_over_sets: Whether to take the mean or the sum of the losses.
        """
        super().__init__()

        self._check_fuzzy_sets(fuzzy_sets)

        self.adaption_losses = nn.ModuleList(adaption_losses)  # registers parameters
        self.fuzzy_sets = fuzzy_sets
        self.mean_over_sets = mean_over_sets

        self.add_state("loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("batch_counter", default=torch.tensor(0), dist_reduce_fx="sum")

    def _check_fuzzy_sets(self, fuzzy_sets: List[Tuple[float, float]]) -> None:
        for i, (left_bound, right_bound) in enumerate(fuzzy_sets):
            if left_bound >= right_bound:
                raise ValueError(
                    f"Fuzzy set {i} with bounds ({left_bound}, {right_bound}) has "
                    f"a left bound greater than or equal to the right bound."
                )

    def update(
        self,
        source: torch.Tensor,
        source_preds: torch.Tensor,
        target: torch.Tensor,
        target_preds: torch.Tensor,
    ) -> None:
        """
        Update the loss with the given data.

        The predictions for the source and target data are checked against the fuzzy
        sets to determine membership.

        Args:
            source: The source features.
            source_preds: The predictions for the source features.
            target: The target features.
            target_preds: The predictions for the target features.
        """
        for fuzzy_set, adaption_loss in zip(self.fuzzy_sets, self.adaption_losses):
            selected_source = source[_membership(source_preds, fuzzy_set)]
            selected_target = target[_membership(target_preds, fuzzy_set)]
            if selected_source.shape[0] > 0 and selected_target.shape[0] > 0:
                self.loss = self.loss + adaption_loss(selected_source, selected_target)
        self.batch_counter += 1

    def compute(self) -> torch.Tensor:
        """
        Compute the loss as either the sum or mean of all subset losses.

        Returns:
            The combined loss.
        """
        if self.batch_counter > 1:
            raise RuntimeError(
                "The update method of this loss was called multiple times before "
                "computing the loss. This is not supported."
            )
        if self.mean_over_sets:
            return self.loss / len(self.adaption_losses)
        else:
            return self.loss


def _membership(preds: torch.Tensor, fuzzy_set: Tuple[float, float]) -> torch.Tensor:
    preds = preds.squeeze() if len(preds.shape) > 1 else preds
    membership = (preds >= fuzzy_set[0]) & (preds < fuzzy_set[1])

    return membership
