from unittest import mock

import pytest
import torch
from torch import nn

import rul_adapt.loss
from rul_adapt.loss.conditional import _membership


@pytest.fixture()
def fuzzy_sets():
    return [(0.0, 0.4), (0.3, 0.7), (0.6, 1.0)]


@pytest.fixture()
def conditional_adaption_loss(fuzzy_sets):
    losses = [
        mock.MagicMock(rul_adapt.loss.MaximumMeanDiscrepancyLoss) for _ in range(3)
    ]
    for loss in losses:
        loss.return_value = torch.tensor(1.0)
    conditional_loss = rul_adapt.loss.ConditionalAdaptionLoss(losses, fuzzy_sets)

    return conditional_loss


@pytest.fixture()
def cdann():
    domain_disc = nn.Linear(10, 1)
    losses = [rul_adapt.loss.DomainAdversarialLoss(domain_disc)]
    conditional_loss = rul_adapt.loss.ConditionalAdaptionLoss(losses, [(0.0, 1.0)])

    return conditional_loss


@pytest.fixture()
def cmmd():
    losses = [rul_adapt.loss.MaximumMeanDiscrepancyLoss(5)]
    conditional_loss = rul_adapt.loss.ConditionalAdaptionLoss(losses, [(0.0, 1.0)])

    return conditional_loss


@pytest.mark.parametrize("faulty_fuzzy_sets", [[(0.1, 0.0)], [(0.1, 0.1)]])
def test_fuzzy_set_boundary_check(faulty_fuzzy_sets):
    with pytest.raises(ValueError):
        rul_adapt.loss.ConditionalAdaptionLoss([mock.MagicMock()], faulty_fuzzy_sets)


def test_loss_aggregation_guard(conditional_adaption_loss):
    inputs = (
        torch.rand(10, 10),
        torch.rand(10, 1),
        torch.rand(10, 10),
        torch.rand(10, 1),
    )

    conditional_adaption_loss.update(*inputs)
    conditional_adaption_loss.update(*inputs)
    with pytest.raises(RuntimeError):
        conditional_adaption_loss.compute()


@mock.patch("rul_adapt.loss.conditional._membership")
def test_update(mock_membership, conditional_adaption_loss, fuzzy_sets):
    source = mock.MagicMock(torch.Tensor)
    source.__getitem__().shape = torch.Size([10, 1])  # set shape of selected samples
    source_preds = mock.MagicMock(torch.Tensor)
    target = mock.MagicMock(torch.Tensor)
    target.__getitem__().shape = torch.Size([10, 1])  # set shape of selected samples
    target_preds = mock.MagicMock(torch.Tensor)

    conditional_adaption_loss.update(source, source_preds, target, target_preds)

    for fuzzy_set in fuzzy_sets:
        mock_membership.assert_any_call(source_preds, fuzzy_set)
        mock_membership.assert_any_call(target_preds, fuzzy_set)
    for loss in conditional_adaption_loss.adaption_losses:
        loss.assert_called()


@pytest.mark.parametrize("mean", [True, False])
def test_compute(conditional_adaption_loss, mean):
    conditional_adaption_loss.mean_over_sets = mean
    conditional_adaption_loss.loss = torch.tensor(3.0)

    computed_loss = conditional_adaption_loss.compute()

    if mean:
        assert computed_loss == torch.tensor(1.0)
    else:
        assert computed_loss == torch.tensor(3.0)


def test_forward(conditional_adaption_loss):
    source = torch.rand(10, 10)
    source_preds = torch.zeros(10, 1)
    target = torch.rand(10, 10)
    target_preds = torch.zeros(10, 1)

    loss = conditional_adaption_loss(source, source_preds, target, target_preds)

    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])
    # all samples fall in first set --> (1 + 0 + 0) / 3
    assert loss == torch.tensor(1 / 3)


def test__membership():
    inputs = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])[:, None]
    fuzzy_set = (0.2, 0.4)
    expected = torch.tensor([False, True, True, False, False])

    assert torch.all(_membership(inputs, fuzzy_set) == expected)


def test_backward_cdann(cdann):
    source = torch.rand(10, 10)
    source_preds = torch.zeros(10, 1)
    target = torch.rand(10, 10)
    target_preds = torch.zeros(10, 1)

    loss = cdann(source, source_preds, target, target_preds)
    loss.backward()

    assert cdann.adaption_losses[0].domain_disc.weight.grad is not None
    assert cdann.adaption_losses[0].domain_disc.bias.grad is not None


def test_backward_cmmd(cmmd):
    source = torch.rand(10, 10, requires_grad=True)
    source_preds = torch.zeros(10, 1)
    target = torch.rand(10, 10, requires_grad=True)
    target_preds = torch.zeros(10, 1)

    loss = cmmd(source, source_preds, target, target_preds)
    loss.backward()

    assert source.grad is not None
    assert target.grad is not None
