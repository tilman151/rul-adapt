from typing import Callable

import torch


def calc_pairwise_euclidean(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return _calc_pairwise_distances(x, y, _euclidean)


def calc_pairwise_dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return _calc_pairwise_distances(x, y, torch.dot)


def _calc_pairwise_distances(
    x: torch.Tensor, y: torch.Tensor, dist_func: Callable
) -> torch.Tensor:
    """Compute pairwise euclidean distances between features."""
    num_elems = x.shape[0]
    x = x.view(num_elems, 1, -1)
    y = y.view(1, num_elems, -1)
    distances = dist_func(x, y)

    return distances


def _euclidean(x, y):
    return torch.sqrt((x - y) ** 2).sum(-1)
