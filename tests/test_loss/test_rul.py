import numpy.testing as npt
import pytest
import torch

from rul_adapt import loss


@pytest.mark.parametrize(
    ["inputs", "targets", "exp_result"],
    [(0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.5, 1.0, 0.3921), (1.5, 1.0, 0.51271)],
)
def test_rul_score_functional(inputs, targets, exp_result):
    inputs = torch.tensor([inputs] * 10)
    targets = torch.tensor([targets] * 10)

    score = loss.rul_score(inputs, targets, pos_factor=10.0, neg_factor=-13.0)

    npt.assert_almost_equal(score, exp_result, decimal=5)


@pytest.mark.parametrize(
    ["inputs", "targets"],
    [(0.0, 0.0), (1.0, 1.0), (0.5, 1.0), (1.5, 1.0)],
)
def test_rul_score_modular(inputs, targets):
    pos_factor = 10.0
    neg_factor = -13.0
    inputs = torch.tensor([inputs] * 50)
    targets = torch.tensor([targets] * 50)
    score_module = loss.RULScore(pos_factor, neg_factor)

    for batch in zip(torch.split(inputs, 10), torch.split(targets, 10)):
        module_score = score_module(*batch)
        functional_score = loss.rul_score(*batch, pos_factor, neg_factor)
        npt.assert_almost_equal(module_score, functional_score)

    total_functional_score = loss.rul_score(inputs, targets, pos_factor, neg_factor)
    total_module_score = score_module.compute()
    npt.assert_almost_equal(total_module_score, total_functional_score)
