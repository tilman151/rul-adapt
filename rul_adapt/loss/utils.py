from typing import Callable, List

import torch


def weighted_mean(
    inputs: List[torch.Tensor], weights: List[torch.Tensor]
) -> torch.Tensor:
    torch_weights = torch.stack(weights).float()
    torch_weights /= torch.sum(torch_weights)
    mean = torch.sum(torch.stack(inputs) * torch_weights)

    return mean


def calc_pairwise_euclidean(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return _calc_pairwise_distances(x, y, _euclidean)


def calc_pairwise_dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return _calc_pairwise_distances(x, y, _dot)


def _calc_pairwise_distances(
    x: torch.Tensor, y: torch.Tensor, dist_func: Callable
) -> torch.Tensor:
    """Compute pairwise euclidean distances between features."""
    x = x[:, None]
    y = y[None, :]
    distances = dist_func(x, y)

    return distances


def _euclidean(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sqrt((x - y) ** 2 + 1e-7).sum(-1)


def _dot(x, y):
    return torch.mul(x, y).sum(-1)
