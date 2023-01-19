from unittest import mock

import numpy as np
import numpy.testing as npt
import pytest
import pytorch_lightning as pl
import rul_datasets
import torch
from torch import nn
from torchmetrics import Metric

from rul_adapt import model
from rul_adapt.approach import ConsistencyApproachPretraining, ConsistencyApproach
from rul_adapt.approach.consistency import StdExtractor


@pytest.fixture()
def pretraining_inputs():
    return torch.randn(10, 14, 20), torch.arange(10, dtype=torch.float)


@pytest.fixture()
def inputs(pretraining_inputs):
    return *pretraining_inputs, torch.randn(10, 14, 20)


@pytest.fixture()
def pretraining_models():
    fe = model.CnnExtractor(14, [10, 5], 20, fc_units=20)
    reg = model.FullyConnectedHead(20, [10, 1], act_func_on_last_layer=False)

    return fe, reg


@pytest.fixture()
def models(pretraining_models):
    disc = model.FullyConnectedHead(20, [1], act_func_on_last_layer=False)

    return *pretraining_models, disc


@pytest.fixture()
def pretraining_approach(pretraining_models):
    approach = ConsistencyApproachPretraining(lr=0.001)
    approach.set_model(*pretraining_models)
    approach.log = mock.MagicMock()

    return approach


@pytest.fixture()
def approach(models):
    approach = ConsistencyApproach(0.1, 0.001, 3000)
    approach.set_model(*models)
    approach.log = mock.MagicMock()

    return approach


@pytest.fixture()
def mocked_pretraining_approach():
    feature_extractor = mock.MagicMock(nn.Module, return_value=torch.zeros(10, 20))
    regressor = mock.MagicMock(nn.Module, return_value=torch.zeros(10, 1))
    approach = ConsistencyApproachPretraining(0.001)
    approach.set_model(feature_extractor, regressor)
    approach.log = mock.MagicMock()

    return approach


@pytest.fixture()
def mocked_approach():
    feature_extractor = mock.MagicMock(nn.Module, return_value=torch.zeros(10, 20))
    regressor = mock.MagicMock(nn.Module, return_value=torch.zeros(10, 1))
    disc = mock.MagicMock(nn.Module, return_value=torch.zeros(10, 1))
    approach = ConsistencyApproach(0.1, 0.001, 3000)
    approach.set_model(feature_extractor, regressor, disc)
    approach.log = mock.MagicMock()

    return approach


class TestConsistencyPretraining:
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

        approach.training_step(pretraining_inputs, batch_idx=0)

        approach.train_loss.assert_called_once()
        approach.log.assert_called_with("train/loss", approach.train_loss)

    @torch.no_grad()
    def test_val_step_logging(self, pretraining_inputs, mocked_pretraining_approach):
        approach = mocked_pretraining_approach
        approach.val_loss = mock.MagicMock(Metric)

        approach.validation_step(pretraining_inputs, batch_idx=0)

        approach.val_loss.assert_called_once()
        approach.log.assert_called_with("val/loss", approach.val_loss)


class TestConsistencyApproach:
    def test_set_model(self, pretraining_approach, approach, models):
        _, _, domain_disc = models
        approach.set_model(
            pretraining_approach.feature_extractor,
            pretraining_approach.regressor,
            domain_disc,
        )

        assert approach.feature_extractor is pretraining_approach.feature_extractor
        assert approach.regressor is pretraining_approach.regressor
        assert hasattr(approach, "dann_loss")  # dann loss was created
        assert approach.dann_loss.domain_disc is domain_disc  # disc was assigned
        assert approach.domain_disc is domain_disc  # domain_disc property works

        param_pairs = zip(
            approach.feature_extractor.parameters(),
            approach.frozen_feature_extractor.parameters(),
        )
        for fe_param, frozen_param in param_pairs:
            assert fe_param is not frozen_param  # frozen params are different objs
            assert torch.dist(fe_param, frozen_param) == 0.0  # have same values
            assert not frozen_param.requires_grad  # are frozen

    def test_domain_disc_check(self, models):
        feature_extractor, regressor, _ = models
        faulty_domain_disc = model.FullyConnectedHead(
            20, [1], act_func_on_last_layer=True
        )
        approach = ConsistencyApproach(1.0, 0.001, 3000)

        with pytest.raises(ValueError):
            approach.set_model(feature_extractor, regressor)

        with pytest.raises(ValueError):
            approach.set_model(feature_extractor, regressor, faulty_domain_disc)

    def test_forward(self, inputs, approach):
        features, *_ = inputs

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

        frozen_extractor_param = next(approach.frozen_feature_extractor.parameters())
        assert frozen_extractor_param.grad is None

    @torch.no_grad()
    def test_val_step(self, inputs, approach):
        features, labels, _ = inputs

        approach.validation_step([features, labels], batch_idx=0, dataloader_idx=0)
        approach.validation_step([features, labels], batch_idx=0, dataloader_idx=1)
        with pytest.raises(RuntimeError):
            approach.validation_step([features, labels], batch_idx=0, dataloader_idx=2)

    @torch.no_grad()
    def test_train_step_logging(self, inputs, mocked_approach):
        approach = mocked_approach
        approach.train_source_loss = mock.MagicMock(Metric)
        approach.dann_loss = mock.MagicMock(Metric)
        approach.consistency_loss = mock.MagicMock(Metric)

        approach.training_step(inputs, batch_idx=0)

        approach.train_source_loss.assert_called_once()
        approach.dann_loss.assert_called_once()
        approach.consistency_loss.assert_called_once()
        approach.log.assert_has_calls(
            [
                mock.call("train/loss", mock.ANY),
                mock.call("train/source_loss", approach.train_source_loss),
                mock.call("train/dann", approach.dann_loss),
                mock.call("train/consistency", approach.consistency_loss),
            ]
        )

    @torch.no_grad()
    def test_val_step_logging(self, mocked_approach, inputs):
        approach = mocked_approach

        features, labels, _ = inputs
        approach.val_source_rmse = mock.MagicMock(Metric)
        approach.val_source_score = mock.MagicMock(Metric)
        approach.val_target_rmse = mock.MagicMock(Metric)
        approach.val_target_score = mock.MagicMock(Metric)

        # check source data loader call
        approach.validation_step([features, labels], batch_idx=0, dataloader_idx=0)
        approach.val_source_rmse.assert_called_once()
        approach.val_source_score.assert_called_once()
        approach.val_target_rmse.assert_not_called()
        approach.val_target_score.assert_not_called()
        approach.log.assert_has_calls(
            [
                mock.call("val/source_rmse", approach.val_source_rmse),
                mock.call("val/source_score", approach.val_source_score),
            ]
        )

        approach.val_source_rmse.reset_mock()
        approach.val_source_score.reset_mock()

        # check target data loader call
        approach.validation_step([features, labels], batch_idx=0, dataloader_idx=1)
        approach.val_source_rmse.assert_not_called()
        approach.val_source_score.assert_not_called()
        approach.val_target_rmse.assert_called_once()
        approach.val_target_score.assert_called_once()
        approach.log.assert_has_calls(
            [
                mock.call("val/target_rmse", approach.val_target_rmse),
                mock.call("val/target_score", approach.val_target_score),
            ]
        )

    @torch.no_grad()
    def test_test_step_logging(self, mocked_approach, inputs):
        approach = mocked_approach
        features, labels, _ = inputs
        approach.test_source_rmse = mock.MagicMock(Metric)
        approach.test_source_score = mock.MagicMock(Metric)
        approach.test_target_rmse = mock.MagicMock(Metric)
        approach.test_target_score = mock.MagicMock(Metric)

        # check source data loader call
        approach.test_step([features, labels], batch_idx=0, dataloader_idx=0)
        approach.test_source_rmse.assert_called_once()
        approach.test_source_score.assert_called_once()
        approach.test_target_rmse.assert_not_called()
        approach.test_target_score.assert_not_called()
        approach.log.assert_has_calls(
            [
                mock.call("test/source_rmse", approach.test_source_rmse),
                mock.call("test/source_score", approach.test_source_score),
            ]
        )

        approach.test_source_rmse.reset_mock()
        approach.test_source_score.reset_mock()

        # check target data loader call
        approach.test_step([features, labels], batch_idx=0, dataloader_idx=1)
        approach.test_source_rmse.assert_not_called()
        approach.test_source_score.assert_not_called()
        approach.test_target_rmse.assert_called_once()
        approach.test_target_score.assert_called_once()
        approach.log.assert_has_calls(
            [
                mock.call("test/target_rmse", approach.test_target_rmse),
                mock.call("test/target_score", approach.test_target_score),
            ]
        )


def test_std_extractor():
    inputs = np.random.randn(100, 3000, 2)
    extractor = StdExtractor(channels=[1])

    outputs = extractor(inputs)

    assert outputs.shape == (100, 1)
    npt.assert_almost_equal(outputs, inputs[:, :, [1]].std(axis=1))


@pytest.mark.integration
def test_on_dummy():
    pl.seed_everything(42)

    fd1 = rul_datasets.reader.DummyReader(fd=1)
    dm = rul_datasets.RulDataModule(fd1, 16)

    feature_extractor = model.CnnExtractor(1, [16], 10, fc_units=10)
    regressor = model.FullyConnectedHead(10, [1], act_func_on_last_layer=False)

    pre_approach = ConsistencyApproachPretraining(0.001)
    pre_approach.set_model(feature_extractor, regressor)

    trainer = pl.Trainer(
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        max_epochs=10,
    )
    trainer.fit(pre_approach, dm)

    fd2 = fd1.get_compatible(fd=2, percent_broken=0.8)
    dm = rul_datasets.DomainAdaptionDataModule(
        rul_datasets.RulDataModule(fd1, 16), rul_datasets.RulDataModule(fd2, 16)
    )

    disc = model.FullyConnectedHead(10, [1], act_func_on_last_layer=False)

    approach = ConsistencyApproach(1.0, 0.001, 10)
    approach.set_model(pre_approach.feature_extractor, pre_approach.regressor, disc)

    trainer = pl.Trainer(
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        max_epochs=10,
    )
    trainer.fit(approach, dm)
    trainer.test(approach, dm)