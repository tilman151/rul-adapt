from unittest import mock

import pytest
import pytorch_lightning as pl
import rul_datasets
import torch
from torch import nn
from torchmetrics import Metric

from rul_adapt import model
from rul_adapt.approach import AdaRulApproachPretraining


@pytest.fixture()
def pretraining_inputs():
    return torch.randn(10, 14, 20), torch.arange(10, dtype=torch.float)


@pytest.fixture()
def pretraining_models():
    fe = model.CnnExtractor(14, [10, 5], 20, fc_units=20)
    reg = model.FullyConnectedHead(20, [10, 1], act_func_on_last_layer=False)

    return fe, reg


@pytest.fixture()
def pretraining_approach(pretraining_models):
    approach = AdaRulApproachPretraining(lr=0.001)
    approach.set_model(*pretraining_models)

    return approach


@pytest.fixture()
def mocked_pretraining_approach():
    feature_extractor = mock.MagicMock(nn.Module, return_value=torch.zeros(10, 20))
    regressor = mock.MagicMock(nn.Module, return_value=torch.zeros(10, 1))
    approach = AdaRulApproachPretraining(0.001)
    approach.set_model(feature_extractor, regressor)

    return approach


class TestAdaRulPretraining:
    def test_forward(self, pretraining_inputs, pretraining_approach):
        features, _ = pretraining_inputs

        outputs = pretraining_approach(features)

        assert outputs.shape == torch.Size([10, 1])

    def test_train_step(self, pretraining_inputs, pretraining_approach):
        outputs = pretraining_approach.training_step(pretraining_inputs, batch_idx=0)

        assert outputs.shape == torch.Size([])

    def test_train_step_backward(self, pretraining_inputs, pretraining_approach):
        outputs = pretraining_approach.training_step(pretraining_inputs, batch_idx=0)
        outputs.backward()

        extractor_parameter = next(pretraining_approach.feature_extractor.parameters())
        assert extractor_parameter.grad is not None

    def test_val_step(self, pretraining_inputs, pretraining_approach):
        pretraining_approach.validation_step(pretraining_inputs, batch_idx=0)

    @torch.no_grad()
    def test_train_step_logging(self, pretraining_inputs, mocked_pretraining_approach):
        approach = mocked_pretraining_approach
        approach.train_loss = mock.MagicMock(Metric)
        approach.log = mock.MagicMock()

        approach.training_step(pretraining_inputs, batch_idx=0)

        approach.train_loss.assert_called_once()
        approach.log.assert_called_with("train/loss", approach.train_loss)

    @torch.no_grad()
    def test_val_step_logging(self, pretraining_inputs, mocked_pretraining_approach):
        approach = mocked_pretraining_approach
        approach.val_loss = mock.MagicMock(Metric)
        approach.log = mock.MagicMock()

        approach.validation_step(pretraining_inputs, batch_idx=0)

        approach.val_loss.assert_called_once()
        approach.log.assert_called_with("val/loss", approach.val_loss)

    @pytest.mark.integration
    def test_on_dummy(self):
        pl.seed_everything(42)

        fd1 = rul_datasets.reader.DummyReader(fd=1)
        dm = rul_datasets.RulDataModule(fd1, 16)

        feature_extractor = model.CnnExtractor(1, [16], 10, fc_units=10)
        regressor = model.FullyConnectedHead(10, [1], act_func_on_last_layer=False)

        approach = AdaRulApproachPretraining(0.001)
        approach.set_model(feature_extractor, regressor)

        trainer = pl.Trainer(
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
            max_epochs=10,
        )
        trainer.fit(approach, dm)
