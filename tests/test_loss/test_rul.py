import numpy.testing as npt
import pytest
import torch

from rul_adapt import loss


@torch.no_grad()
@pytest.mark.parametrize(
    ["inputs", "targets", "exp_result"],
    [(0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.5, 1.0, 0.3921), (1.5, 1.0, 0.51271)],
)
def test_rul_score_functional(inputs, targets, exp_result):
    inputs = torch.tensor([inputs] * 10)
    targets = torch.tensor([targets] * 10)

    score = loss.rul_score(
        inputs,
        targets,
        pos_factor=10.0,
        neg_factor=-13.0,
        offset=-1.0,
        as_percentage=False,
    )

    npt.assert_almost_equal(score, exp_result, decimal=5)


@pytest.mark.parametrize("as_percentage", [True, False])
def test_rul_score_functional_backward(as_percentage):
    inputs = torch.randn(10, 1, requires_grad=True)
    targets = torch.randn(10, 1)

    score = loss.rul_score(
        inputs,
        targets,
        pos_factor=10.0,
        neg_factor=-13.0,
        offset=-1.0,
        as_percentage=as_percentage,
    )
    score.backward()

    assert inputs.grad is not None


@torch.no_grad()
@pytest.mark.parametrize(
    ["inputs", "targets"],
    [(0.0, 0.0), (1.0, 1.0), (0.5, 1.0), (1.5, 1.0)],
)
@pytest.mark.parametrize("mode", ["phm08", "phm12"])
def test_rul_score_modular(inputs, targets, mode):
    inputs = torch.tensor([inputs] * 52) + torch.randn(52) * 0.0001
    targets = torch.tensor([targets] * 52)
    score_module = loss.RULScore(mode)

    # split into batches of 10 with a last batch of 2 to test proper averaging
    for batch in zip(torch.split(inputs, 10), torch.split(targets, 10)):
        module_score = score_module(*batch)
        functional_score = loss.rul_score(
            *batch,
            score_module.pos_factor,
            score_module.neg_factor,
            score_module.offset,
            score_module.as_percentage
        )
        if mode == "phm12":
            functional_score /= batch[0].shape[0]
        npt.assert_almost_equal(module_score, functional_score)

    total_functional_score = loss.rul_score(
        inputs,
        targets,
        score_module.pos_factor,
        score_module.neg_factor,
        score_module.offset,
        score_module.as_percentage,
    )
    if mode == "phm12":
        total_functional_score /= inputs.shape[0]
    total_module_score = score_module.compute()
    npt.assert_almost_equal(total_module_score, total_functional_score, decimal=5)


@pytest.mark.parametrize(["mode", "default"], [("phm08", False), ("phm12", True)])
def test_mean_override(mode, default):
    score_module = loss.RULScore(mode)
    assert score_module.mean == default

    for mean in [True, False]:
        score_module = loss.RULScore(mode, mean=mean)
        assert score_module.mean == mean


def test_rul_score_modular_backward():
    inputs = torch.randn(10, 1, requires_grad=True)
    targets = torch.randn(10, 1)
    score_module = loss.RULScore()

    score = score_module(inputs, targets)
    score.backward()

    assert inputs.grad is not None
