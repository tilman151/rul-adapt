import math

import numpy.testing as npt
import pytest
import torch

from rul_adapt import loss


@pytest.fixture(autouse=True)
def manual_seed():
    torch.manual_seed(42)


def test_healthy_state_alignment():
    inputs = torch.randn(1000, 2)
    inputs[:, 0] *= 2
    inputs[:, 1] *= 4
    healthy_state_align = loss.HealthyStateAlignmentLoss()

    healthy_loss = healthy_state_align(inputs)

    npt.assert_almost_equal(healthy_loss.item(), (2**2 + 4**2) / 2, decimal=0)


@pytest.mark.parametrize(
    ["inputs", "expected"],
    [
        ([[1.0, 2.0], [4.0, 4.0]], -0.5 * (1.0 + math.cos(math.pi / 4))),  # 90°
        ([[1.0, 2.0], [4.0, 1.0]], -0.5 * (1.0 + math.cos(math.pi / 2))),  # 45°
        ([[0.0, 0.0], [4.0, 4.0]], 0.0),  # 180°
        ([[1.0, 1.0], [1.0, 1.0]], 0.0),  # direction is zero
    ],
)
def test_direction_alignment(inputs, expected):
    """
    Loss is mean of normalized, pair-wise dot:
        * 2 times dot prod with self -> 1.
        * 2 times dot between vectors -> cos(angle between vectors)
    """
    healthy = torch.ones(1, 2)
    degraded = torch.tensor(inputs)
    direction_align = loss.DegradationDirectionAlignmentLoss()

    direction_loss = direction_align(healthy, degraded)

    npt.assert_almost_equal(direction_loss.item(), expected)


@pytest.mark.parametrize(
    ["source_degradation_steps", "target_degradation_steps", "expected"],
    [
        ([10.0, 20.0], [[10.0, 20.0]], 0.0),  # perfect alignment
        ([20.0, 10.0], [[10.0, 20.0]], 0.5),  # source 0.5 before and behind
        ([10.0, 20.0], [[20.0, 10.0]], 0.5),  # target 0.5 before and behind
    ],
)
def test_degradation_regularization(
    source_degradation_steps, target_degradation_steps, expected
):
    healthy = torch.ones(1, 2)
    source = torch.tensor([[1.0, 2.0], [1.0, 3.0]])
    source_labels = torch.tensor(source_degradation_steps)
    target = torch.tensor([[1.0, 2.0], [1.0, 3.0]])
    target_degradation_steps = torch.tensor(target_degradation_steps)
    degradation_reg = loss.DegradationLevelRegularizationLoss()

    reg_loss = degradation_reg(
        healthy, source, source_labels, target, target_degradation_steps
    )

    npt.assert_almost_equal(reg_loss, expected)


def test_degradation_regularization_no_source_difference():
    healthy = torch.ones(1, 2)
    source = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    source_labels = torch.tensor([10.0, 20.0])
    target = torch.tensor([[1.0, 2.0], [1.0, 3.0]])
    target_degradation_steps = torch.tensor([10.0, 20.0])
    degradation_reg = loss.DegradationLevelRegularizationLoss()

    reg_loss = degradation_reg(
        healthy, source, source_labels, target, target_degradation_steps
    )

    assert not torch.isnan(reg_loss)


@pytest.mark.parametrize(
    ["metric", "inputs"],
    [
        (loss.HealthyStateAlignmentLoss(), (torch.randn(10, 5),)),
        (
            loss.DegradationDirectionAlignmentLoss(),
            (torch.randn(10, 5), torch.randn(10, 5)),
        ),
        (
            loss.DegradationLevelRegularizationLoss(),
            (
                torch.randn(10, 5),
                torch.randn(10, 5),
                torch.arange(10, dtype=torch.float),
                torch.randn(10, 5),
                torch.arange(10, dtype=torch.float),
            ),
        ),
    ],
)
def test_aggregated_metrics_same_magnitude(metric, inputs):
    """Check if batched aggregates have at least approximately the same magnitude as
    directly computed values."""
    batched_inputs = (torch.split(i, 6) for i in inputs)  # batches of 6 and 5 samples
    batched_inputs = zip(*batched_inputs)

    outputs = metric(*inputs)
    metric.reset()

    for i in batched_inputs:
        metric(*i)
    batched_outputs = metric.compute()

    npt.assert_almost_equal(outputs, batched_outputs, decimal=0)
