from typing import Iterable
from unittest import mock

import numpy.testing as npt
import pytest
import torch
from torch import nn

from rul_adapt import loss


@pytest.fixture(autouse=True)
def manual_seed():
    torch.manual_seed(42)


def test_mmd_same_dist():
    source = torch.randn(1000, 10)
    target = torch.randn(1000, 10)
    mmd = loss.MaximumMeanDiscrepancyLoss(num_kernels=5)

    mmd_loss = mmd(source, target)

    npt.assert_almost_equal(mmd_loss.item(), 0.0, decimal=3)


@pytest.mark.parametrize("source_batch_size", [100, 500])
@pytest.mark.parametrize("target_batch_size", [100, 500])
def test_mmd_diff_dist(source_batch_size, target_batch_size):
    source = torch.randn(source_batch_size, 10)
    target = torch.randn(target_batch_size, 10)
    mmd = loss.MaximumMeanDiscrepancyLoss(num_kernels=5)

    mmd_loss_1 = mmd(source * 2, target)
    mmd_loss_2 = mmd((source * 2) + 1, target)

    assert mmd_loss_1 < mmd_loss_2


def test_mmd_backward():
    source = torch.randn(100, 10, requires_grad=True)
    target = torch.randn(100, 10, requires_grad=True)
    mmd = loss.MaximumMeanDiscrepancyLoss(num_kernels=5)

    mmd_loss_1 = mmd(source * 2, target)
    mmd_loss_1.backward()

    assert not source.grad.isnan().any()
    assert not target.grad.isnan().any()


def test_jmmd_same_dist():
    source = [torch.randn(1000, 1) for _ in range(2)]
    target = [torch.randn(1000, 1) for _ in range(2)]
    jmmd = loss.JointMaximumMeanDiscrepancyLoss()

    jmmd_loss = jmmd(source, target)

    npt.assert_almost_equal(jmmd_loss.item(), 0.0, decimal=3)


@pytest.mark.parametrize("source_batch_size", [100, 500])
@pytest.mark.parametrize("target_batch_size", [100, 500])
def test_jmmd_diff_dist(source_batch_size, target_batch_size):
    source1 = [2 * torch.randn(source_batch_size, 1) for _ in range(2)]
    source2 = [(2 * torch.randn(source_batch_size, 1)) + 1 for _ in range(2)]
    target = [torch.randn(target_batch_size, 1) for _ in range(2)]
    jmmd = loss.JointMaximumMeanDiscrepancyLoss()

    mmd_loss_1 = jmmd(source1, target)
    mmd_loss_2 = jmmd(source2, target)

    assert mmd_loss_1 < mmd_loss_2


def test_dann_perfect_disc():
    source = torch.randn(50, 10) - 50
    target = torch.randn(50, 10) + 50
    dummy_disc = lambda x: torch.mean(x, dim=1, keepdim=True)
    dann = loss.DomainAdversarialLoss(dummy_disc)

    dann_loss = dann(source, target)

    npt.assert_almost_equal(dann_loss.item(), 0.0, decimal=3)


def test_dann_backward():
    source = torch.randn(50, 10)
    target = torch.randn(50, 10)
    dummy_disc = nn.Linear(10, 1)
    dann = loss.DomainAdversarialLoss(dummy_disc)

    dann_loss = dann(source, target)
    dann_loss.backward()

    assert dummy_disc.weight.grad is not None
    assert dummy_disc.bias.grad is not None


@mock.patch("rul_adapt.loss.adaption.GradientReversalLayer.forward")
def test_dann_grl(mock_grl):
    source = torch.randn(1, 1)
    target = torch.randn(1, 1)
    preds = torch.zeros(2, 1)
    mock_disc = mock.MagicMock(return_value=preds)
    dann = loss.DomainAdversarialLoss(mock_disc)

    dann(source, target)

    mock_grl.assert_called()  # GRL was called
    mock_disc.assert_called_with(mock_grl())  # disc received GRL outputs


@pytest.mark.parametrize("grad_weight", [1.0, 2.0])
def test_gradient_reversal_layer(grad_weight):
    inputs = torch.randn(1, requires_grad=True)
    grl = loss.adaption.GradientReversalLayer(grad_weight)

    inputs.mean().backward()
    grad = inputs.grad  # regular gradient

    inputs.grad = None  # reset gradient
    grl(inputs).mean().backward()
    grad_grl = inputs.grad  # gradient passed through GRL

    assert -grad_weight * grad == grad_grl


@pytest.mark.parametrize(
    ["leader", "follower", "expected"],
    [
        (torch.ones(16, 2), torch.ones(16, 2) * 3, 2.0),
        (torch.ones(16, 2), torch.ones(16, 2), 0.0),
    ],
)
def test_consistency_loss(leader, follower, expected):
    metric = loss.ConsistencyLoss()

    consistency_loss = metric(leader, follower)

    assert isinstance(consistency_loss, torch.Tensor)
    assert consistency_loss.shape == torch.Size([])
    assert consistency_loss == expected


@pytest.mark.parametrize(
    ["metric", "inputs"],
    [
        (
            loss.DomainAdversarialLoss(lambda x: torch.mean(x, dim=1, keepdim=True)),
            (torch.randn(10, 1), torch.randn(10, 1)),
        ),
        (
            loss.MaximumMeanDiscrepancyLoss(5),
            (torch.randn(10, 1), torch.randn(10, 1)),
        ),
        (
            loss.JointMaximumMeanDiscrepancyLoss(),
            ([torch.randn(10, 1)], [torch.randn(10, 1)]),
        ),
        (
            loss.ConsistencyLoss(),
            (torch.randn(10, 1), torch.randn(10, 1)),
        ),
    ],
)
class TestMetricsWithDevices:
    def test_device_moving(self, metric, inputs):
        """Regression: check if metric can be moved from one device to another."""
        metric(*inputs)
        metric.cpu()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU was detected")
    def test_on_gpu(self, metric, inputs):
        """Regression: check if metric can be moved from one device to another."""

        def _to(movable):
            if isinstance(movable, (list, tuple)):
                return [_to(m) for m in movable]
            else:
                return movable.to("cuda:0")

        metric = metric.to("cuda:0")
        metric(*_to(inputs))
        metric.compute()
