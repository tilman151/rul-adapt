from unittest import mock

import pytest
import pytorch_lightning as pl
import rul_datasets
import torch
from torch import nn

from rul_adapt import model
from rul_adapt.approach import SupervisedApproach, AdaRulApproach
from tests.test_approach import utils


@pytest.fixture()
def inputs():
    return (
        torch.randn(10, 14, 20),
        torch.arange(10, dtype=torch.float),
        torch.randn(10, 14, 20),
    )


@pytest.fixture()
def models():
    fe = model.LstmExtractor(14, [32, 32, 32], bidirectional=True)
    reg = model.FullyConnectedHead(64, [32, 1], act_func_on_last_layer=False)
    disc = model.FullyConnectedHead(64, [32, 1], act_func_on_last_layer=False)

    return fe, reg, disc


@pytest.fixture()
def approach(models, mocker):
    approach = AdaRulApproach(130, 35, 1, lr=0.01)
    approach.set_model(*models)
    approach.log = mock.MagicMock()

    approach.manual_backward = mock.MagicMock(name="manual_backward")
    approach.optimizers = mock.MagicMock(
        name="optimizers",
        return_value=(
            mock.MagicMock(name="disc_optim"),
            mock.MagicMock(name="gen_optim"),
        ),
    )

    return approach


class TestAdaRulApproach:
    def test_set_model(self, approach, models):
        feature_extractor, regressor, domain_disc = models
        approach.set_model(feature_extractor, regressor, domain_disc)

        assert approach.feature_extractor is feature_extractor
        assert approach.regressor is regressor
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
        approach = AdaRulApproach(130, 35, 1, lr=0.01)

        with pytest.raises(ValueError):
            approach.set_model(feature_extractor, regressor)

        with pytest.raises(ValueError):
            approach.set_model(feature_extractor, regressor, faulty_domain_disc)

    def test_forward(self, inputs, approach):
        features, *_ = inputs

        outputs = approach(features)

        assert outputs.shape == torch.Size([10, 1])

    def test_train_step(self, inputs, approach):
        mock_disc_loss = mock.MagicMock(
            name="disc_loss", return_value=torch.zeros(10, requires_grad=True)
        )
        mock_gen_loss = mock.MagicMock(
            name="gen_loss", return_value=torch.zeros(10, requires_grad=True)
        )
        approach._get_disc_loss = mock_disc_loss
        approach._get_gen_loss = mock_gen_loss

        for e in range(2):
            approach.on_train_epoch_start()  # reset counters on start of epoch
            for i in range(100):
                approach.training_step(inputs, batch_idx=i)
                # start over after all updates are done
                i = i % (approach.num_disc_updates + approach.num_gen_updates)
                if i < approach.num_disc_updates:  # make disc updates first
                    mock_disc_loss.assert_called()
                    mock_gen_loss.assert_not_called()
                else:  # make gen updates afterwards
                    mock_disc_loss.assert_not_called()
                    mock_gen_loss.assert_called()
                mock_disc_loss.reset_mock()
                mock_gen_loss.reset_mock()

    def test_optimizer_configured_with_factory(self, mocker, models):
        mock_factory = mocker.patch("rul_adapt.utils.OptimizerFactory")
        kwargs = {"optim_type": "sgd", "lr": 0.001, "weight_decay": 0.001}
        approach = AdaRulApproach(125, 10, 10, **kwargs)
        approach.set_model(*models)
        approach.configure_optimizers()

        mock_factory.assert_called_once_with(**kwargs)
        mock_factory().assert_called()

    def test_train_step_backward_disc(self, inputs, approach):
        """Feature extractor should have no gradients in disc step. Disc should have
        gradients. Frozen feature extractor should never have gradients."""
        approach.training_step(inputs, batch_idx=0)

        approach.manual_backward.assert_called()
        disc_optim, _ = approach.optimizers()
        disc_optim.step.assert_called()

    def test_train_step_backward_gen(self, inputs, approach):
        """Feature extractor should have gradients in gen step. Disc will necessarily
        have gradients, but this is unimportant here."""
        approach._disc_counter = approach.num_disc_updates
        approach.training_step(inputs, batch_idx=0)

        approach.manual_backward.assert_called()
        _, gen_optim = approach.optimizers()
        gen_optim.step.assert_called()

    @torch.no_grad()
    def test_val_step(self, inputs, approach):
        features, labels, _ = inputs

        approach.validation_step([features, labels], batch_idx=0, dataloader_idx=0)
        approach.validation_step([features, labels], batch_idx=0, dataloader_idx=1)
        with pytest.raises(RuntimeError):
            approach.validation_step([features, labels], batch_idx=0, dataloader_idx=2)

    @torch.no_grad()
    def test_train_step_disc_logging(self, inputs, approach):
        approach.gan_loss = mock.MagicMock(nn.Module)
        approach._disc_counter = 0

        approach.training_step(inputs, batch_idx=0)

        approach.gan_loss.assert_called_once()
        approach.log.assert_called_with("train/disc_loss", approach.gan_loss())

    @torch.no_grad()
    def test_train_step_gen_logging(self, inputs, approach):
        approach.gan_loss = mock.MagicMock(nn.Module)
        approach._disc_counter = approach.num_disc_updates
        approach._gen_counter = 0

        approach.training_step(inputs, batch_idx=0)

        approach.gan_loss.assert_called_once()
        approach.log.assert_called_with("train/gen_loss", approach.gan_loss())

    @torch.no_grad()
    def test_val_step_logging(self, approach, mocker):
        utils.check_val_logging(approach, mocker)

    @torch.no_grad()
    def test_test_step_logging(self, approach, mocker):
        utils.check_test_logging(approach, mocker)

    def test_model_hparams_logged(self, models, mocker):
        approach = AdaRulApproach(125, 1, 1)
        mocker.patch.object(approach, "log_model_hyperparameters")

        approach.set_model(*models)

        approach.log_model_hyperparameters.assert_called_with("_domain_disc")

    def test_checkpointing(self, tmp_path):
        ckpt_path = tmp_path / "checkpoint.ckpt"
        fe = model.ActivationDropoutWrapper(
            model.CnnExtractor(1, [16], 10, fc_units=16), nn.ReLU, 0.5
        )
        reg = model.FullyConnectedHead(16, [1])
        disc = model.FullyConnectedHead(16, [1], act_func_on_last_layer=False)
        approach = AdaRulApproach(130, 35, 1, lr=0.01)
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

    pre_approach = SupervisedApproach("mse", 130)
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

    approach = AdaRulApproach(130, 35, 1, lr=0.0001)
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
