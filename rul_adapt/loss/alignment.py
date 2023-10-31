"""These losses are used to create a latent space that is conductive to RUL
estimation. They are mainly used by the [LatentAlignmentApproach]
[rul_adapt.approach.LatentAlignApproach]."""

from typing import List

import torch
import torchmetrics

from rul_adapt.loss.utils import calc_pairwise_dot, weighted_mean


class HealthyStateAlignmentLoss(torchmetrics.Metric):
    """
    This loss is used to align the healthy state of the data in the latent space.

    It computes the mean of the variance of each latent feature which is supposed to
    be minimized. This way a single compact cluster of healthy state data should be
    formed.

    The loss is implemented as a [torchmetrics.Metric](
    https://torchmetrics.readthedocs.io/en/stable/pages/quickstart.html#module
    -metrics). See their documentation for more information.

    Examples:
        ```pycon
        >>> from rul_adapt.loss.alignment import HealthyStateAlignmentLoss
        >>> healthy_align = HealthyStateAlignmentLoss()
        >>> healthy_align(torch.zeros(10, 5))
        tensor(0.)
        ```
    """

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False

    loss: List[torch.Tensor]
    total: List[torch.Tensor]

    def __init__(self):
        super().__init__()

        self.add_state("loss", [], dist_reduce_fx="cat")
        self.add_state("total", [], dist_reduce_fx="cat")

    def update(self, healthy: torch.Tensor) -> None:
        self.loss.append(torch.mean(torch.var(healthy, dim=0)))
        self.total.append(torch.tensor(healthy.shape[0], device=self.device))

    def compute(self) -> torch.Tensor:
        return weighted_mean(self.loss, self.total)


class DegradationDirectionAlignmentLoss(torchmetrics.Metric):
    """
    This loss is used to align the direction of the degradation data in relation to
    the healthy state data in the latent space.

    It computes the mean of the cosine of the pairwise-angle of the vectors from the
    healthy state cluster to each degradation data point. The healthy state cluster
    location is assumed to be the mean of the healthy state data in the latent space.
    The loss is negated in order to maximize the cosine by minimizing the loss.

    The loss is implemented as a [torchmetrics.Metric](
    https://torchmetrics.readthedocs.io/en/stable/pages/quickstart.html#module
    -metrics). See their documentation for more information.

    Examples:
        ```pycon
        >>> from rul_adapt.loss.alignment import DegradationDirectionAlignmentLoss
        >>> degradation_align = DegradationDirectionAlignmentLoss()
        >>> degradation_align(torch.zeros(10, 5), torch.ones(10, 5))
        tensor(-1.0000)
        ```
    """

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False

    loss: torch.Tensor
    total: torch.Tensor

    def __init__(self):
        super().__init__()

        self.add_state("loss", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, healthy: torch.Tensor, degraded: torch.Tensor) -> None:
        healthy_mean = healthy.mean(dim=0)
        trajectory = degraded - healthy_mean
        trajectory_norm = torch.norm(trajectory, dim=1, keepdim=True)
        trajectory = trajectory / (trajectory_norm + 1e-8)
        pairwise_dist = calc_pairwise_dot(trajectory, trajectory)
        loss = -pairwise_dist.sum()

        self.loss = self.loss + loss
        self.total = self.total + (degraded.shape[0] ** 2)

    def compute(self) -> torch.Tensor:
        return self.loss / self.total


class DegradationLevelRegularizationLoss(torchmetrics.Metric):
    """
    This loss is used to regularize the degradation level of the data in the latent
    space in relation to the healthy state data.

    It computes the mean of the difference between the normalized distance of the
    degradation data points from the healthy state cluster and the normalized
    degradation steps. The healthy state cluster location is assumed to be the mean
    of the healthy state data in the latent space.

    The loss is implemented as a [torchmetrics.Metric](
    https://torchmetrics.readthedocs.io/en/stable/pages/quickstart.html#module
    -metrics). See their documentation for more information.

    Examples:
        ```pycon
        >>> from rul_adapt.loss.alignment import DegradationLevelRegularizationLoss
        >>> degradation_align = DegradationLevelRegularizationLoss()
        >>> degradation_align(torch.zeros(10, 5),
        ...                   torch.ones(10, 5),
        ...                   torch.ones(10),
        ...                   torch.ones(10, 5),
        ...                   torch.ones(10))
        tensor(0.)
        ```
    """

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False

    loss: torch.Tensor
    total: torch.Tensor

    def __init__(self) -> None:
        super().__init__()

        self.add_state("loss", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        healthy: torch.Tensor,
        source: torch.Tensor,
        source_degradation_steps: torch.Tensor,
        target: torch.Tensor,
        target_degradation_steps: torch.Tensor,
    ) -> None:
        healthy_mean = healthy.mean(dim=0)
        source_distances = self._calc_normed_distances(healthy_mean, source)
        target_distances = self._calc_normed_distances(healthy_mean, target)

        source_max = torch.max(source_degradation_steps)
        source_degradation_steps = source_degradation_steps / source_max
        target_max = torch.max(target_degradation_steps)
        target_degradation_steps = target_degradation_steps / target_max

        source_loss = torch.abs(source_distances - source_degradation_steps).sum()
        target_loss = torch.abs(target_distances - target_degradation_steps).sum()
        loss = source_loss + target_loss

        self.loss = self.loss + loss
        self.total = self.total + source.shape[0]

    def compute(self) -> torch.Tensor:
        return self.loss / self.total

    def _calc_normed_distances(
        self, healthy_mean: torch.Tensor, source: torch.Tensor
    ) -> torch.Tensor:
        distances = torch.norm(source - healthy_mean, p=2, dim=1)
        distances = distances / (torch.max(distances) + 1e-8)

        return distances
