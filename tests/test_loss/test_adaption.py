from unittest import mock

import numpy.testing as npt
import pytest
import torch

from rul_adapt import loss


@pytest.fixture(autouse=True)
def manual_seed():
    torch.manual_seed(42)


def test_mmd_same_dist():
    source = torch.randn(1000, 1)
    target = torch.randn(1000, 1)
    mmd = loss.MaximumMeanDiscrepancyLoss(num_kernels=5)

    mmd_loss = mmd(source, target)

    npt.assert_almost_equal(mmd_loss.item(), 0.0, decimal=3)


def test_mmd_diff_dist():
    source = torch.randn(1000, 1)
    target = torch.randn(1000, 1)
    mmd = loss.MaximumMeanDiscrepancyLoss(num_kernels=5)

    mmd_loss_1 = mmd(source * 2, target)
    mmd_loss_2 = mmd((source * 2) + 1, target)

    assert mmd_loss_1 < mmd_loss_2


def test_jmmd_same_dist():
    source = [torch.randn(1000, 1) for _ in range(2)]
    target = [torch.randn(1000, 1) for _ in range(2)]
    jmmd = loss.JointMaximumMeanDiscrepancyLoss()

    jmmd_loss = jmmd(source, target)

    npt.assert_almost_equal(jmmd_loss.item(), 0.0, decimal=3)


def test_jmmd_diff_dist():
    source1 = [2 * torch.randn(1000, 1) for _ in range(2)]
    source2 = [(2 * torch.randn(1000, 1)) + 1 for _ in range(2)]
    target = [torch.randn(1000, 1) for _ in range(2)]
    jmmd = loss.JointMaximumMeanDiscrepancyLoss()

    mmd_loss_1 = jmmd(source1, target)
    mmd_loss_2 = jmmd(source2, target)

    assert mmd_loss_1 < mmd_loss_2


def test_dann_perfect_disc():
    source = torch.randn(500, 1) - 50
    target = torch.randn(500, 1) + 50
    dummy_disc = lambda x: torch.mean(x, dim=1, keepdim=True)
    dann = loss.DomainAdversarialLoss(dummy_disc)

    dann_loss = dann(source, target)

    npt.assert_almost_equal(dann_loss.item(), 0.0, decimal=3)


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


def test_gradient_reversal_layer():
    inputs = torch.randn(1, requires_grad=True)
    grl = loss.adaption.GradientReversalLayer()

    inputs.mean().backward()
    grad = inputs.grad  # regular gradient

    inputs.grad = None  # reset gradient
    grl(inputs).mean().backward()
    grad_grl = inputs.grad  # gradient passed through GRL

    assert -grad == grad_grl
