from typing import Any

import torch
import torchmetrics


class RULScore(torchmetrics.Metric):
    is_differentiable = True
    higher_is_better = False
    full_state_update = False

    loss: torch.Tensor
    total: torch.Tensor

    def __init__(self, pos_factor=10.0, neg_factor=-13.0, mean: bool = False):
        super().__init__()
        self.pos_factor = pos_factor
        self.neg_factor = neg_factor
        self.mean = mean

        self.add_state("loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        self.loss += rul_score(inputs, targets, self.pos_factor, self.neg_factor)
        self.total += inputs.shape[0]

    def compute(self) -> Any:
        if self.mean:
            return self.loss / self.total
        else:
            return self.loss


def rul_score(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    pos_factor: float,
    neg_factor: float,
) -> torch.Tensor:
    dist = inputs - targets
    for i, d in enumerate(dist):
        dist[i] = (d / neg_factor) if d < 0 else (d / pos_factor)
    dist = torch.exp(dist) - 1
    score = dist.sum()

    return score
