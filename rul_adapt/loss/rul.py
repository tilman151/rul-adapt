import math
from typing import Any, Literal, Optional

import torch
import torchmetrics


class RULScore(torchmetrics.Metric):
    is_differentiable = True
    higher_is_better = None
    full_state_update = False

    loss: torch.Tensor
    total: torch.Tensor

    def __init__(
        self, mode: Literal["phm08", "phm12"] = "phm08", mean: Optional[bool] = None
    ):
        super().__init__()
        if mode == "phm08":
            self.pos_factor = 10.0
            self.neg_factor = -13.0
            self.offset = -1.0
            self.as_percentage = False
            self.mean = False
        elif mode == "phm12":
            self.pos_factor = 20 / math.log(0.5)
            self.neg_factor = -5 / math.log(0.5)
            self.offset = 0.0
            self.as_percentage = True
            self.mean = True
        else:
            raise ValueError(f"Unknown RUL score mode: {mode}")

        if mean is not None:
            self.mean = mean

        self.add_state("loss", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        self.loss = self.loss + rul_score(
            inputs,
            targets,
            self.pos_factor,
            self.neg_factor,
            self.offset,
            self.as_percentage,
        )
        self.total = self.total + inputs.shape[0]

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
    offset: float,
    as_percentage: bool,
) -> torch.Tensor:
    dist = inputs - targets
    if as_percentage:
        dist = dist / targets * 100
    factors = torch.ones_like(dist)
    factors[dist >= 0] /= pos_factor
    factors[dist < 0] /= neg_factor
    dist = torch.exp(dist * factors) + offset
    score = dist.sum()

    return score
