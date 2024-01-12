from typing import Callable
from unittest import mock

import pytest
import pytorch_lightning as pl
import torch
import torchmetrics
import numpy.testing as npt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

import rul_adapt.loss
from rul_adapt.approach.evaluation import AdaptionEvaluator


class DummyModule(pl.LightningModule):
    def __init__(self, model):
        super().__init__()

        self.model = model

        self.evaluator = AdaptionEvaluator(self.forward, self.log)

    def forward(self, x):
        return self.model(x)

    def validation_step(self, batch, batch_idx, dataloader_idx):
        domain = "source" if dataloader_idx == 0 else "target"
        self.evaluator.validation(batch, domain)

    def test_step(self, batch, batch_idx, dataloader_idx):
        domain = "source" if dataloader_idx == 0 else "target"
        self.evaluator.test(batch, domain)


@pytest.fixture()
def lightning_module():
    return DummyModule(lambda x: torch.flatten(x, start_dim=1).mean((1,), True))


@pytest.fixture()
def mocked_evaluator(domain, prefix, mocker):
    mock_forward = mocker.Mock(nn.Module, return_value=torch.randn(10, 1))
    mock_log = mocker.Mock(Callable[[str, torchmetrics.Metric], None])
    evaluator = AdaptionEvaluator(mock_forward, mock_log)
    setattr(
        evaluator,
        f"{prefix}_metrics",
        nn.ModuleDict(
            {
                domain: nn.ModuleDict(
                    {
                        "rmse": mock.Mock(torchmetrics.Metric),
                        "score": mock.Mock(torchmetrics.Metric),
                    }
                )
            }
        ),
    )

    return evaluator


@pytest.mark.parametrize(
    ["step_func", "prefix"], [("validation", "val"), ("test", "test")]
)
@pytest.mark.parametrize("domain", ["source", "target"])
def test_step(step_func, prefix, domain, mocked_evaluator):
    batch = [torch.randn(10, 3, 5), torch.randn(10)]

    getattr(mocked_evaluator, step_func)(batch, domain)
    mocked_evaluator.network_func.assert_called_once()
    mocked_evaluator.log_func.assert_has_calls(
        [
            mock.call(
                f"{prefix}/{domain}/rmse",
                getattr(mocked_evaluator, f"{prefix}_metrics")[domain]["rmse"],
            ),
            mock.call(
                f"{prefix}/{domain}/score",
                getattr(mocked_evaluator, f"{prefix}_metrics")[domain]["score"],
            ),
        ]
    )


@pytest.mark.parametrize("eval_func", ["validate", "test"])
def test_metric_aggregation(lightning_module, eval_func):
    source = torch.randn(105, 3, 5), torch.randn(105)
    target = torch.randn(105, 3, 5), torch.randn(105)
    source_dl = DataLoader(TensorDataset(*source), batch_size=10)
    target_dl = DataLoader(TensorDataset(*target), batch_size=10)
    trainer = pl.Trainer(logger=False)
    mse = torchmetrics.MeanSquaredError(squared=False)
    rul_score = rul_adapt.loss.RULScore()

    for _ in range(3):  # run multiple times to check that metrics are reset
        results = getattr(trainer, eval_func)(lightning_module, [source_dl, target_dl])
        iterator = enumerate(zip([source, target], results))
        for i, ((features, targets), eval_metrics) in iterator:
            preds = lightning_module.forward(features)
            targets = targets[:, None]
            exp_rmse = mse(preds, targets).item()
            exp_score = rul_score(preds, targets).item()
            actual_rmse, actual_score = eval_metrics.values()

            npt.assert_almost_equal(exp_rmse, actual_rmse, decimal=5)
            npt.assert_almost_equal(exp_score, actual_score, decimal=5)


@pytest.mark.parametrize(
    ["step_func", "prefix"], [("validation", "val"), ("test", "test")]
)
@pytest.mark.parametrize("domain", ["source", "target"])
def test_degraded_only_evaluation(mocked_evaluator, domain, prefix, step_func):
    mocked_evaluator.degraded_only = True
    healthy_lables = torch.ones(5)
    healthy_features = torch.ones(5, 3, 5)
    degraded_labels = torch.rand(5)  # smaller than 1.0
    degraded_features = torch.zeros(5, 3, 5)
    batch = [
        torch.cat([healthy_features, degraded_features]),
        torch.cat([healthy_lables, degraded_labels]),
    ]

    getattr(mocked_evaluator, step_func)(batch, domain)

    (actual_network_input,) = mocked_evaluator.network_func.call_args.args
    assert torch.dist(actual_network_input, degraded_features) == 0.0
