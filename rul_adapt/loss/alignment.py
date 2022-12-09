import torch
import torchmetrics

from rul_adapt.loss.utils import calc_pairwise_dot


class HealthyStateAlignmentLoss(torchmetrics.Metric):
    """TODO: implement running var
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance"""

    def __init__(self):
        super().__init__()

        self.add_state("loss", [], dist_reduce_fx="cat")
        self.add_state("total", [], dist_reduce_fx="cat")

    def update(self, healthy: torch.Tensor) -> None:
        self.loss.append(torch.mean(torch.var(healthy, dim=0)))
        self.total.append(healthy.shape[0])

    def compute(self) -> torch.Tensor:
        weights = torch.tensor(self.total, device=self.device, dtype=torch.float)
        weights /= torch.sum(weights)
        loss = torch.sum(torch.stack(self.loss) * weights)

        return loss


class DegradationDirectionAlignmentLoss(torchmetrics.Metric):
    def __init__(self):
        super().__init__()

        self.add_state("loss", [], dist_reduce_fx="cat")
        self.add_state("total", [], dist_reduce_fx="cat")

    def update(self, healthy: torch.Tensor, degraded: torch.Tensor) -> None:
        healthy_mean = healthy.mean(dim=0)
        trajectory = degraded - healthy_mean
        trajectory = trajectory / torch.norm(trajectory, dim=1)[:, None]
        pairwise_dist = calc_pairwise_dot(trajectory, trajectory)
        loss = -pairwise_dist.mean()

        self.loss.append(loss)
        self.total.append(degraded.shape[0])

    def compute(self) -> torch.Tensor:
        weights = torch.tensor(self.total, device=self.device, dtype=torch.float)
        weights /= torch.sum(weights)
        loss = torch.sum(torch.stack(self.loss) * weights)

        return loss


class DegradationLevelRegularizationLoss(torchmetrics.Metric):
    def __init__(self, max_rul: int) -> None:
        super().__init__()

        self.max_rul = max_rul

        self.add_state("loss", [], dist_reduce_fx="cat")
        self.add_state("total", [], dist_reduce_fx="cat")

    def update(
        self,
        healthy: torch.Tensor,
        source: torch.Tensor,
        source_labels: torch.Tensor,
        target: torch.Tensor,
        target_degradation_steps: torch.Tensor,
    ) -> None:
        healthy_mean = healthy.mean(dim=0)
        source_distances = self._calc_normed_distances(healthy_mean, source)
        target_distances = self._calc_normed_distances(healthy_mean, target)

        source_degradation_steps = self.max_rul - source_labels
        source_degradation_steps /= torch.max(source_degradation_steps)
        target_degradation_steps /= torch.max(target_degradation_steps)

        source_loss = torch.abs(source_distances - source_degradation_steps).mean()
        target_loss = torch.abs(target_distances - target_degradation_steps).mean()
        loss = source_loss + target_loss

        self.loss.append(loss)
        self.total.append(source.shape[0])

    def compute(self) -> torch.Tensor:
        weights = torch.tensor(self.total, device=self.device, dtype=torch.float)
        weights /= torch.sum(weights)
        loss = torch.sum(torch.stack(self.loss) * weights)

        return loss

    def _calc_normed_distances(
        self, healthy_mean: torch.Tensor, source: torch.Tensor
    ) -> torch.Tensor:
        distances = torch.norm(source - healthy_mean, p=2, dim=1)
        distances /= torch.max(distances)

        return distances
