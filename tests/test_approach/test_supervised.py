from unittest import mock

import pytest
import torch
import torchmetrics
from torchmetrics import Metric

from rul_adapt import model
from rul_adapt.approach import SupervisedApproach


@pytest.fixture()
def inputs():
    return torch.randn(10, 14, 20), torch.arange(10, dtype=torch.float)


@pytest.fixture()
def models():
    fe = model.LstmExtractor(14, [32, 32, 32], bidirectional=True)
    reg = model.FullyConnectedHead(64, [32, 1], act_func_on_last_layer=False)

    return fe, reg


@pytest.fixture()
def approach(models):
    approach = SupervisedApproach(
        lr=0.001, loss_type="mse", optim_type="adam", rul_scale=130
    )
    approach.set_model(*models)

    return approach


class TestSupervisedApproach:
    @pytest.mark.parametrize(
        ["loss_type", "exp_loss", "squared"],
        [
            ("mse", torchmetrics.MeanSquaredError, True),
            ("rmse", torchmetrics.MeanSquaredError, False),
            ("mae", torchmetrics.MeanAbsoluteError, None),
        ],
    )
    def test_loss_selection(self, loss_type, exp_loss, squared):
        supervised = SupervisedApproach(0.1, loss_type, "adam")

        assert isinstance(supervised.train_loss, exp_loss)
        if squared is not None:
            assert supervised.train_loss.squared == squared

    @pytest.mark.parametrize(
        ["optim_type", "exp_optim"],
        [("adam", torch.optim.Adam), ("sgd", torch.optim.SGD)],
    )
    def test_configure_optimizers(self, models, optim_type, exp_optim):
        supervised = SupervisedApproach(0.1, "mse", optim_type)
        supervised.set_model(*models)
        optim = supervised.configure_optimizers()

        assert isinstance(optim, exp_optim)
        assert optim.param_groups[0]["params"] == list(supervised.parameters())

    def test_forward(self, inputs, approach):
        features, _ = inputs

        outputs = approach(features)

        assert outputs.shape == torch.Size([10, 1])

    def test_train_step(self, inputs, approach):
        outputs = approach.training_step(inputs, batch_idx=0)

        assert outputs.shape == torch.Size([])

    def test_train_step_backward(self, inputs, approach):
        outputs = approach.training_step(inputs, batch_idx=0)
        outputs.backward()

        extractor_parameter = next(approach.feature_extractor.parameters())
        assert extractor_parameter.grad is not None

    def test_val_step(self, inputs, approach):
        approach.validation_step(inputs, batch_idx=0)

    @torch.no_grad()
    def test_train_step_logging(self, inputs, approach):
        approach = approach

        approach.train_loss = mock.MagicMock(Metric)
        approach.log = mock.MagicMock()

        approach.training_step(inputs, batch_idx=0)

        approach.train_loss.assert_called_once()
        approach.log.assert_called_with("train/loss", approach.train_loss)

    @torch.no_grad()
    def test_val_step_logging(self, inputs, approach):
        approach = approach

        approach.val_loss = mock.MagicMock(Metric)
        approach.log = mock.MagicMock()

        approach.validation_step(inputs, batch_idx=0)

        approach.val_loss.assert_called_once()
        approach.log.assert_called_with("val/loss", approach.val_loss)
