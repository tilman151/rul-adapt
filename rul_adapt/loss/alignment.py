import torch
from torch import nn

from rul_adapt.loss.utils import calc_pairwise_dot


class HealthyStateAlignmentLoss(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.var(inputs, dim=0))


class DegradationDirectionAlignmentLoss(nn.Module):
    def forward(self, healthy: torch.Tensor, degraded: torch.Tensor) -> torch.Tensor:
        healthy_mean = healthy.mean(dim=0)
        trajectory = degraded - healthy_mean
        trajectory = trajectory / torch.norm(trajectory, dim=1)[:, None]
        pairwise_dist = calc_pairwise_dot(trajectory, trajectory)
        loss = -pairwise_dist.mean()

        return loss


class DegradationLevelRegularizationLoss(nn.Module):
    def __init__(self, max_rul: int) -> None:
        super().__init__()

        self.max_rul = max_rul

    def forward(
        self,
        healthy: torch.Tensor,
        source: torch.Tensor,
        source_labels: torch.Tensor,
        target: torch.Tensor,
        target_degradation_steps: torch.Tensor,
    ) -> torch.Tensor:
        healthy_mean = healthy.mean(dim=0)
        source_distances = self._calc_normed_distances(healthy_mean, source)
        target_distances = self._calc_normed_distances(healthy_mean, target)

        source_degradation_steps = self.max_rul - source_labels
        source_degradation_steps /= torch.max(source_degradation_steps)
        target_degradation_steps /= torch.max(target_degradation_steps)

        source_loss = torch.abs(source_distances - source_degradation_steps).mean()
        target_loss = torch.abs(target_distances - target_degradation_steps).mean()
        loss = source_loss + target_loss

        return loss

    def _calc_normed_distances(
        self, healthy_mean: torch.Tensor, source: torch.Tensor
    ) -> torch.Tensor:
        distances = torch.norm(source - healthy_mean, p=2, dim=1)
        distances /= torch.max(distances)

        return distances
