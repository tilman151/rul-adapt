from unittest import mock

import numpy as np
import numpy.testing as npt
import pytest
import pytorch_lightning as pl
import rul_datasets
import torch
from torchmetrics import Metric

from rul_adapt import model
from rul_adapt.approach import ConsistencyApproach, SupervisedApproach
from rul_adapt.approach.consistency import StdExtractor, TumblingWindowExtractor
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
    fe = model.CnnExtractor(14, [10, 5], 20, fc_units=20)
    reg = model.FullyConnectedHead(20, [10, 1], act_func_on_last_layer=False)
    disc = model.FullyConnectedHead(20, [1], act_func_on_last_layer=False)

    return fe, reg, disc


@pytest.fixture()
def approach(models):
    approach = ConsistencyApproach(0.1, 3000, lr=0.001)
    approach.set_model(*models)
    approach.log = mock.MagicMock()

    return approach


class TestConsistencyApproach:
    def test_set_model(self, approach, models):
        feature_extractor, regressor, domain_disc = models
        approach.set_model(feature_extractor, regressor, domain_disc)

        assert approach.feature_extractor is feature_extractor
        assert approach.regressor is regressor
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
        approach = ConsistencyApproach(1.0, 3000, lr=0.001)

        with pytest.raises(ValueError):
            approach.set_model(feature_extractor, regressor)

        with pytest.raises(ValueError):
            approach.set_model(feature_extractor, regressor, faulty_domain_disc)

    def test_optimizer_configured_with_factory(self, models, mocker):
        mock_factory = mocker.patch("rul_adapt.utils.OptimizerFactory")
        kwargs = {"optim_type": "sgd", "lr": 0.001, "weight_decay": 0.001}
        approach = ConsistencyApproach(1.0, 3000, **kwargs)
        approach.set_model(*models)
        approach.configure_optimizers()

        mock_factory.assert_called_once_with(**kwargs)
        mock_factory().assert_called_once()

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
    def test_train_step_logging(self, inputs, approach):
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
    def test_val_step_logging(self, approach, mocker):
        utils.check_val_logging(approach, mocker)

    @torch.no_grad()
    def test_test_step_logging(self, approach, mocker):
        utils.check_test_logging(approach, mocker)

    def test_checkpointing(self, tmp_path):
        ckpt_path = tmp_path / "checkpoint.ckpt"
        fe = model.CnnExtractor(1, [16], 10, fc_units=16)
        reg = model.FullyConnectedHead(16, [1])
        disc = model.FullyConnectedHead(16, [1], act_func_on_last_layer=False)
        approach = ConsistencyApproach(1.0, 1, lr=0.001)
        approach.set_model(fe, reg, disc)

        utils.checkpoint(approach, ckpt_path)
        ConsistencyApproach.load_from_checkpoint(ckpt_path)


def test_std_extractor():
    inputs = np.random.randn(100, 3000, 2)
    targets = np.random.randn(100)
    extractor = StdExtractor(channels=[1])

    ex_inputs, ex_targets = extractor(inputs, targets)

    assert ex_inputs.shape == (100, 1)
    npt.assert_almost_equal(ex_inputs, inputs[:, :, [1]].std(axis=1))
    assert ex_targets.shape == targets.shape
    npt.assert_equal(ex_targets, targets)


def test_tumbling_window_extractor():
    inputs = np.random.randn(100, 3000, 2)
    targets = np.arange(len(inputs), 0, -1)
    extractor = TumblingWindowExtractor(30, [0])

    ex_inputs, ex_targets = extractor(inputs, targets)

    assert ex_inputs.shape == (100 * 100, 30, 1)
    npt.assert_equal(inputs[10, 30:60, 0:1], ex_inputs[10 * 100 + 1])  # check window

    assert ex_targets.shape == (100 * 100,)
    assert targets[10] == ex_targets[10 * 100 + 1]


@pytest.mark.integration
def test_on_dummy():
    pl.seed_everything(42)

    fd1 = rul_datasets.reader.DummyReader(fd=1)
    dm = rul_datasets.RulDataModule(fd1, 16)

    feature_extractor = model.CnnExtractor(1, [16], 10, fc_units=10)
    regressor = model.FullyConnectedHead(10, [1], act_func_on_last_layer=False)

    pre_approach = SupervisedApproach(0.001, "mse", "sgd")
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

    approach = ConsistencyApproach(1.0, 10, lr=0.001)
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
