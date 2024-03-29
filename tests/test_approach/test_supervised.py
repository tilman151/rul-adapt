from unittest import mock

import pytest
import pytorch_lightning as pl
import rul_datasets.reader
import torch
import torchmetrics
from torchmetrics import Metric

from rul_adapt import model
from rul_adapt.approach import SupervisedApproach
from tests.test_approach import utils


@pytest.fixture()
def inputs():
    return [torch.randn(10, 1, 20), torch.arange(10, dtype=torch.float)]


@pytest.fixture()
def models():
    fe = model.LstmExtractor(1, [32, 32, 32], bidirectional=True)
    reg = model.FullyConnectedHead(64, [32, 1], act_func_on_last_layer=False)

    return fe, reg


@pytest.fixture()
def approach(models):
    approach = SupervisedApproach(loss_type="mse", rul_scale=130)
    approach.set_model(*models)

    return approach


@pytest.fixture()
def dummy_dm():
    source = rul_datasets.RulDataModule(rul_datasets.reader.DummyReader(1), 32)
    target = rul_datasets.RulDataModule(rul_datasets.reader.DummyReader(2), 32)
    dm = rul_datasets.DomainAdaptionDataModule(source, target)

    return dm


@pytest.fixture()
def silent_trainer():
    return pl.Trainer(
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
    )


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
        supervised = SupervisedApproach(loss_type)

        assert isinstance(supervised.train_loss, exp_loss)
        if squared is not None:
            assert supervised.train_loss.squared == squared

    def test_optimizer_configured_with_factory(self, models, mocker):
        mock_factory = mocker.patch("rul_adapt.utils.OptimizerFactory")
        kwargs = {"optim_type": "sgd", "lr": 0.001, "weight_decay": 0.001}
        approach = SupervisedApproach("mse", **kwargs)
        approach.configure_optimizers()

        mock_factory.assert_called_once_with(**kwargs)
        mock_factory().assert_called_once()

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

    @torch.no_grad()
    def test_test_step(self, approach, inputs):
        approach.test_step(inputs, batch_idx=0, dataloader_idx=0)
        approach.test_step(inputs, batch_idx=0, dataloader_idx=1)
        with pytest.raises(RuntimeError):
            approach.test_step(inputs, batch_idx=0, dataloader_idx=2)

    @torch.no_grad()
    def test_test_step_logging(self, approach, mocker):
        utils.check_test_logging(approach, mocker)

    def test_validation_switcher(self, approach, dummy_dm, silent_trainer, mocker):
        mock_no_adapt_val = mocker.patch.object(approach, "_no_adapt_validation_step")
        mock_adapt_val = mocker.patch.object(approach.evaluator, "validation")

        silent_trainer.validate(approach, dummy_dm.source)
        mock_no_adapt_val.assert_called()
        mock_adapt_val.assert_not_called()

        silent_trainer.validate(approach, dummy_dm)
        mock_adapt_val.assert_called()

    def test_test_switcher(self, approach, dummy_dm, silent_trainer, mocker):
        mock_no_adapt_test = mocker.patch.object(approach, "_no_adapt_test_step")
        mock_adapt_test = mocker.patch.object(approach.evaluator, "test")

        silent_trainer.test(approach, dummy_dm.source)
        mock_no_adapt_test.assert_called()
        mock_adapt_test.assert_not_called()

        silent_trainer.test(approach, dummy_dm)
        mock_adapt_test.assert_called()
