from unittest import mock

import pytest
import pytorch_lightning as pl
import rul_datasets
import torch
from torch import nn
from torchmetrics import Metric

from rul_adapt import model
from rul_adapt.approach import AdaRulApproachPretraining, AdaRulApproach
from tests.test_approach import utils


@pytest.fixture()
def pretraining_inputs():
    return torch.randn(10, 14, 20), torch.arange(10, dtype=torch.float)


@pytest.fixture()
def inputs(pretraining_inputs):
    return *pretraining_inputs, torch.randn(10, 14, 20)


@pytest.fixture()
def pretraining_models():
    fe = model.LstmExtractor(14, [32, 32, 32], bidirectional=True)
    reg = model.FullyConnectedHead(64, [32, 1], act_func_on_last_layer=False)

    return fe, reg


@pytest.fixture()
def models(pretraining_models):
    disc = model.FullyConnectedHead(64, [32, 1], act_func_on_last_layer=False)

    return *pretraining_models, disc


@pytest.fixture()
def pretraining_approach(pretraining_models):
    approach = AdaRulApproachPretraining(lr=0.001, max_rul=130)
    approach.set_model(*pretraining_models)

    return approach


@pytest.fixture()
def approach(models):
    approach = AdaRulApproach(0.001, max_rul=130)
    approach.set_model(*models)
    approach.log = mock.MagicMock()

    return approach


@pytest.fixture()
def mocked_pretraining_approach():
    feature_extractor = mock.MagicMock(nn.Module, return_value=torch.zeros(10, 20))
    regressor = mock.MagicMock(nn.Module, return_value=torch.zeros(10, 1))
    approach = AdaRulApproachPretraining(0.001, 130)
    approach.set_model(feature_extractor, regressor)

    return approach


@pytest.fixture()
def mocked_approach():
    feature_extractor = mock.MagicMock(nn.Module, return_value=torch.zeros(10, 20))
    regressor = mock.MagicMock(nn.Module, return_value=torch.zeros(10, 1))
    disc = mock.MagicMock(nn.Module, return_value=torch.zeros(10, 1))
    approach = AdaRulApproach(0.001, 130)
    approach.set_model(feature_extractor, regressor, disc)
    approach.log = mock.MagicMock()

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


class TestAdaRulApproach:
    def test_set_model(self, pretraining_approach, approach, models):
        _, _, domain_disc = models
        approach.set_model(
            pretraining_approach.feature_extractor,
            pretraining_approach.regressor,
            domain_disc,
        )

        assert approach.feature_extractor is pretraining_approach.feature_extractor
        assert approach.regressor is pretraining_approach.regressor
        assert hasattr(approach, "_domain_disc")  # domain_disc was assigned
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
        approach = AdaRulApproach(0.01, 130)

        with pytest.raises(ValueError):
            approach.set_model(feature_extractor, regressor)

        with pytest.raises(ValueError):
            approach.set_model(feature_extractor, regressor, faulty_domain_disc)

    def test_forward(self, inputs, approach):
        features, *_ = inputs

        outputs = approach(features)

        assert outputs.shape == torch.Size([10, 1])

    def test_train_step(self, inputs, approach):
        outputs = approach.training_step(inputs, batch_idx=0, optimizer_idx=0)
        assert outputs.shape == torch.Size([])

        outputs = approach.training_step(inputs, batch_idx=0, optimizer_idx=1)
        assert outputs.shape == torch.Size([])

        with pytest.raises(RuntimeError):
            approach.training_step(inputs, batch_idx=0, optimizer_idx=2)

    def test_configure_optimizer(self, approach):
        disc_optim, gen_optim = approach.configure_optimizers()

        assert isinstance(disc_optim, torch.optim.Adam)
        assert disc_optim.param_groups[0]["params"] == list(
            approach.domain_disc.parameters()
        )

        assert isinstance(gen_optim, torch.optim.Adam)
        assert gen_optim.param_groups[0]["params"] == list(
            approach.feature_extractor.parameters()
        )

    def test_train_step_backward_disc(self, inputs, approach):
        """Feature extractor should have no gradients in disc step. Disc should have
        gradients. Frozen feature extractor should never have gradients."""
        outputs = approach.training_step(inputs, batch_idx=0, optimizer_idx=0)
        outputs.backward()

        for param in approach.feature_extractor.parameters():
            assert param.grad is None
        for param in approach.frozen_feature_extractor.parameters():
            assert param.grad is None
        for param in approach.domain_disc.parameters():
            assert param.grad is not None

    def test_train_step_backward_gen(self, inputs, approach):
        """Feature extractor should have gradients in gen step. Disc will necessarily
        have gradients but this is unimportant here."""
        outputs = approach.training_step(inputs, batch_idx=0, optimizer_idx=1)
        outputs.backward()

        for param in approach.feature_extractor.parameters():
            assert param.grad is not None

    @torch.no_grad()
    def test_val_step(self, inputs, approach):
        features, labels, _ = inputs

        approach.validation_step([features, labels], batch_idx=0, dataloader_idx=0)
        approach.validation_step([features, labels], batch_idx=0, dataloader_idx=1)
        with pytest.raises(RuntimeError):
            approach.validation_step([features, labels], batch_idx=0, dataloader_idx=2)

    @torch.no_grad()
    def test_train_step_disc_logging(self, inputs, mocked_approach):
        approach = mocked_approach
        approach.gan_loss = mock.MagicMock(nn.Module)

        approach.training_step(inputs, batch_idx=0, optimizer_idx=0)

        approach.gan_loss.assert_called_once()
        approach.log.assert_called_with("train/disc_loss", approach.gan_loss())

    @torch.no_grad()
    def test_train_step_gen_logging(self, inputs, mocked_approach):
        approach = mocked_approach
        approach.gan_loss = mock.MagicMock(nn.Module)

        approach.training_step(inputs, batch_idx=0, optimizer_idx=1)

        approach.gan_loss.assert_called_once()
        approach.log.assert_called_with("train/gen_loss", approach.gan_loss())

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

    def test_checkpointing(self, tmp_path):
        ckpt_path = tmp_path / "checkpoint.ckpt"
        fe = model.ActivationDropoutWrapper(
            model.CnnExtractor(1, [16], 10, fc_units=16), nn.ReLU, 0.5
        )
        reg = model.FullyConnectedHead(16, [1])
        disc = model.FullyConnectedHead(16, [1], act_func_on_last_layer=False)
        approach = AdaRulApproach(0.01, 130)
        approach.set_model(fe, reg, disc)

        utils.checkpoint(approach, ckpt_path, max_rul=130)
        AdaRulApproach.load_from_checkpoint(ckpt_path)


@pytest.mark.integration
def test_on_dummy():
    pl.seed_everything(42)

    fd1 = rul_datasets.reader.DummyReader(fd=1)
    dm = rul_datasets.RulDataModule(fd1, 16)

    feature_extractor = model.LstmExtractor(1, [10], bidirectional=True)
    regressor = model.FullyConnectedHead(20, [1], act_func_on_last_layer=False)

    pre_approach = AdaRulApproachPretraining(0.0001, 130)
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

    disc = model.FullyConnectedHead(20, [1], act_func_on_last_layer=False)

    approach = AdaRulApproach(0.0001, 130)
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
