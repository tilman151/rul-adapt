"""A module for unsupervised domain adaption losses."""

from typing import List, Any

import torch
import torchmetrics
from torch import nn

from rul_adapt.loss.utils import calc_pairwise_euclidean, weighted_mean


class MaximumMeanDiscrepancyLoss(torchmetrics.Metric):
    """The Maximum Mean Discrepancy Loss (MMD) is a distance measure between two
    arbitrary distributions.

    The distance is defined as the dot product in a reproducing Hilbert kernel space
    (RHKS) and is zero if and only if the distributions are identical. The RHKS is
    the space of the linear combination of multiple Gaussian kernels with bandwidths
    derived by the median heuristic.

    The source and target feature batches are treated as samples from their
    respective distribution. The linear pairwise distances between the two batches
    are transformed into distances in the RHKS via the kernel trick:

    ```python
    rhks(x, y) = dot(to_rhks(x), to_rhks(y)) = multi_kernel(dot(x, y))
    multi_kernel(distance) = mean([gaussian(distance, bw) for bw in bandwidths])
    gaussian(distance, bandwidth) = exp(-distance * bandwidth)
    ```

    The n kernels will use bandwidths between `median / (2**(n/ 2))` and `median * (
    2**(n / 2))`, where `median` is the median of the linear distances.

    The MMD loss is then calculated as:

    ```python
    mean(rhks(source, source) + rhks(target, target) - 2 * rhks(source, target))
    ```
    """

    is_differentiable = True
    higher_is_better = False
    full_state_update = False

    loss: List[torch.Tensor]
    total: List[torch.Tensor]

    def __init__(self, num_kernels: int) -> None:
        """
        Create a new MMD loss module with `n` kernels.

        The bandwidths of the Gaussian kernels are derived by the median heuristic.

        Args:
            num_kernels: Number of Gaussian kernels to use.
        """
        super().__init__()

        self.num_kernels = num_kernels

        self.add_state("loss", [], dist_reduce_fx="cat")
        self.add_state("total", [], dist_reduce_fx="cat")

    def update(
        self, source_features: torch.Tensor, target_features: torch.Tensor
    ) -> None:
        """
        Compute the MMD loss between source and target feature distributions.

        Args:
            source_features: Source features with shape `[batch_size, num_features]`
            target_features: Target features with shape `[batch_size, num_features]`

        Returns:
            scalar MMD loss
        """
        feats = torch.cat([source_features, target_features], dim=0)
        distances = calc_pairwise_euclidean(feats, feats)

        gammas = _get_gammas(distances, self.num_kernels)
        distances = _calc_multi_kernel(distances, gammas)

        batch_size = source_features.shape[0]
        disc = _calc_discrepancy(distances, batch_size)

        self.loss.append(disc)
        self.total.append(torch.tensor(batch_size, device=self.device))

    def compute(self) -> torch.Tensor:
        return weighted_mean(self.loss, self.total)


class JointMaximumMeanDiscrepancyLoss(torchmetrics.Metric):
    """The Joint Maximum Mean Discrepancy Loss (JMMD) is a distance measure between
    multiple pairs of arbitrary distributions.

    It is related to the MMD insofar as the distance of each distribution pair in a
    reproducing Hilbert kernel space (RHKS) is calculated and then multiplied before
    the discrepancy is computed.

    ```python
    joint_rhks(xs, ys) = prod(rhks(x, y) for x, y in zip(xs, xs))
    ```

    For more information see
    [MaximumMeanDiscrepancyLoss] [rul_adapt.loss.adaption.MaximumMeanDiscrepancyLoss].
    """

    is_differentiable = True
    higher_is_better = False
    full_state_update = False

    loss: List[torch.Tensor]
    total: List[torch.Tensor]

    def __init__(self) -> None:
        """
        Create a new JMMD loss module.

        It features a single Gaussian kernel with a bandwidth chosen by the median
        heuristic.
        """
        super().__init__()

        self.add_state("loss", [], dist_reduce_fx="cat")
        self.add_state("total", [], dist_reduce_fx="cat")

    def update(
        self, source_features: List[torch.Tensor], target_features: List[torch.Tensor]
    ) -> None:
        """
        Compute the JMMD loss between multiple feature distributions.

        Args:
            source_features: The list of source features of shape
                             `[batch_size, num_features]`.
            target_features: The list of target features of shape
                             `[batch_size, num_features]`.

        Returns:
            scalar JMMD loss
        """
        distances = []
        for source, target in zip(source_features, target_features):
            feats = torch.cat([source, target], dim=0)
            dist = calc_pairwise_euclidean(feats, feats)
            (gamma,) = _get_gammas(dist, 1)
            distances.append(_calc_gaussian_kernel(dist, gamma))

        batch_size = source_features[0].shape[0]
        merged_distances = torch.stack(distances, dim=0).prod(dim=0)
        disc = _calc_discrepancy(merged_distances, batch_size)

        self.loss.append(disc)
        self.total.append(torch.tensor(batch_size, device=self.device))

    def compute(self) -> torch.Tensor:
        return weighted_mean(self.loss, self.total)


class DomainAdversarialLoss(torchmetrics.Metric):
    """The Domain Adversarial Neural Network Loss (DANN) uses a domain discriminator
    to measure the distance between two feature distributions.

    The domain discriminator is a neural network that is jointly trained on
    classifying its input as one of two domains. Its output should be a single
    unscaled score (logit) which is fed to a binary cross entropy loss.

    The domain discriminator is preceded by a [GradientReversalLayer]
    [rul_adapt.loss.adaption.GradientReversalLayer]. This way, the discriminator is
    trained to separate the domains while the network generating the inputs is
    trained to marginalize the domain difference."""

    is_differentiable = True
    higher_is_better = False
    full_state_update = False

    loss: List[torch.Tensor]
    total: List[torch.Tensor]

    def __init__(self, domain_disc: nn.Module) -> None:
        """
        Create a new DANN loss module.

        Args:
            domain_disc: The neural network to act as the domain discriminator.
        """
        super().__init__()

        self.domain_disc = domain_disc
        self.grl = GradientReversalLayer()

        self.add_state("loss", [], dist_reduce_fx="cat")
        self.add_state("total", [], dist_reduce_fx="cat")

    def update(self, source: torch.Tensor, target: torch.Tensor) -> None:
        """
        Calculate the DANN loss as the binary cross entropy of the discriminators
        prediction for the source and target features.

        The source features receive a domain label of zero and the target features a
        domain label of one.

        Args:
            source: The source features with domain label zero.
            target: The target features with domain label one.
        """
        inputs = torch.cat([source, target])
        combined_batch_size = inputs.shape[0]

        labels = torch.ones(combined_batch_size, 1, device=self.device)
        source_batch_size = source.shape[0]
        labels[:source_batch_size] *= 0.0

        predictions = self.domain_disc(self.grl(inputs))
        loss = nn.functional.binary_cross_entropy_with_logits(predictions, labels)

        self.loss.append(loss)
        self.total.append(torch.tensor(combined_batch_size, device=self.device))

    def compute(self) -> torch.Tensor:
        return weighted_mean(self.loss, self.total)


class GradientReversalLayer(nn.Module):
    """The gradient reversal layer (GRL) acts as an identity function in the forward
    pass and negates the gradient in the backward pass.

    ```python
    GRL(f(x)) = f(x)
    GRL`(f(x)) = -f`(x)
    ```
    """

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return _gradient_reversal(inputs)


def _gradient_reversal(x: torch.Tensor) -> torch.Tensor:
    """Perform gradient reversal on input."""
    return _GradientReverse.apply(x)


class _GradientReverse(torch.autograd.Function):
    """Gradient reversal forward and backward definitions."""

    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Forward pass as identity mapping."""
        return inputs

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Backward pass as negative of gradient."""
        return -grad

    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        pass


def _calc_discrepancy(distances: torch.Tensor, batch_size: int) -> torch.Tensor:
    s2s = distances[:batch_size, :batch_size]
    t2t = distances[batch_size:, batch_size:]
    s2t = distances[:batch_size, batch_size:]
    disc = torch.mean(s2s + t2t - 2 * s2t)

    return disc


def _calc_multi_kernel(distances: torch.Tensor, gammas: List[float]) -> torch.Tensor:
    """Compute linear combination of Gaussian kernels."""
    kernels = [_calc_gaussian_kernel(distances, gamma) for gamma in gammas]
    combined: torch.Tensor = sum(kernels) / len(gammas)  # type: ignore

    return combined


def _calc_gaussian_kernel(distances: torch.Tensor, gamma: float) -> torch.Tensor:
    """Compute a single Gaussian kernel."""
    return torch.exp(-distances * gamma)


def _get_gammas(distances: torch.Tensor, num_kernels: int) -> List[float]:
    """Compute gammas for n Gaussian kernels via median heuristic."""
    n_samples = distances.shape[0]
    bandwidth = (n_samples**2 - n_samples) / torch.sum(distances.detach())
    bandwidth /= 2 ** (num_kernels // 2)
    gammas = [bandwidth * (2**i) for i in range(num_kernels)]

    return gammas
