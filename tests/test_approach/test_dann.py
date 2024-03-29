from unittest import mock

import pytest
import rul_datasets.reader
import torch
import torchmetrics
import pytorch_lightning as pl
from torchmetrics import Metric

import rul_adapt.loss
from rul_adapt import model
from rul_adapt.approach.dann import DannApproach
from tests.test_approach import utils


@pytest.fixture()
def models():
    fe = model.LstmExtractor(14, [16], 4)
    reg = model.FullyConnectedHead(4, [1], act_func_on_last_layer=False)
    disc = model.FullyConnectedHead(4, [1], act_func_on_last_layer=False)

    return fe, reg, disc


@pytest.fixture()
def approach(models):
    feature_extractor, regressor, domain_disc = models
    approach = DannApproach(1.0, lr=0.001)
    approach.set_model(feature_extractor, regressor, domain_disc)

    return approach


@pytest.fixture()
def inputs():
    return (
        torch.randn(10, 14, 30),  # source
        torch.arange(10, dtype=torch.float),  # source_labels
        torch.randn(10, 14, 30),  # target
    )


def test_set_model(models):
    feature_extractor, regressor, domain_disc = models
    approach = DannApproach(1.0, lr=0.001)
    approach.set_model(feature_extractor, regressor, domain_disc)

    assert approach.feature_extractor is feature_extractor
    assert approach.regressor is regressor
    assert hasattr(approach, "dann_loss")  # dann loss was created
    assert approach.dann_loss.domain_disc is domain_disc  # disc was assigned correctly
    assert approach.domain_disc is domain_disc  # domain_disc property works


def test_domain_disc_check(models):
    feature_extractor, regressor, _ = models
    faulty_domain_disc = model.FullyConnectedHead(4, [1], act_func_on_last_layer=True)
    approach = DannApproach(1.0, lr=0.001)

    with pytest.raises(ValueError):
        approach.set_model(feature_extractor, regressor)

    with pytest.raises(ValueError):
        approach.set_model(feature_extractor, regressor, faulty_domain_disc)


@pytest.mark.parametrize(
    ["loss_type", "expected"],
    [
        ("mae", torchmetrics.MeanAbsoluteError()),
        ("mse", torchmetrics.MeanSquaredError()),
        ("rmse", torchmetrics.MeanSquaredError(squared=False)),
    ],
)
def test_loss_type(loss_type, expected):
    approach = DannApproach(1.0, lr=0.001, loss_type=loss_type)

    assert approach.loss_type == loss_type
    assert approach.train_source_loss == expected


def test_optimizer_configured_with_factory(models, mocker):
    mock_factory = mocker.patch("rul_adapt.utils.OptimizerFactory")
    kwargs = {"optim_type": "sgd", "lr": 0.001, "weight_decay": 0.001}
    approach = DannApproach(1.0, **kwargs)
    approach.configure_optimizers()

    mock_factory.assert_called_once_with(**kwargs)
    mock_factory().assert_called_once()


@torch.no_grad()
def test_forward(approach, inputs):
    features, _, _ = inputs

    outputs = approach(features)

    assert outputs.shape == torch.Size([10, 1])


@torch.no_grad()
def test_train_step(approach, inputs):

    outputs = approach.training_step(inputs, batch_idx=0)

    assert outputs.shape == torch.Size([])


def test_train_step_backward(approach, inputs):

    outputs = approach.training_step(inputs, batch_idx=0)
    outputs.backward()

    extractor_parameter = next(approach.feature_extractor.parameters())
    assert extractor_parameter.grad is not None


@torch.no_grad()
def test_val_step(approach, inputs):
    features, labels, _ = inputs

    approach.validation_step([features, labels], batch_idx=0, dataloader_idx=0)
    approach.validation_step([features, labels], batch_idx=0, dataloader_idx=1)
    with pytest.raises(RuntimeError):
        approach.validation_step([features, labels], batch_idx=0, dataloader_idx=2)


@torch.no_grad()
def test_test_step(approach, inputs):
    features, labels, _ = inputs

    approach.test_step([features, labels], batch_idx=0, dataloader_idx=0)
    approach.test_step([features, labels], batch_idx=0, dataloader_idx=1)
    with pytest.raises(RuntimeError):
        approach.test_step([features, labels], batch_idx=0, dataloader_idx=2)


@torch.no_grad()
def test_train_step_logging(approach, inputs):
    approach.train_source_loss = mock.MagicMock(Metric)
    approach.dann_loss = mock.MagicMock(rul_adapt.loss.DomainAdversarialLoss)
    approach.log = mock.MagicMock()

    approach.training_step(inputs, batch_idx=0)

    approach.train_source_loss.assert_called_once()
    approach.dann_loss.assert_called_once()

    approach.log.assert_has_calls(
        [
            mock.call("train/loss", mock.ANY),
            mock.call("train/source_loss", approach.train_source_loss),
            mock.call("train/dann", approach.dann_loss),
        ]
    )


@torch.no_grad()
def test_val_step_logging(approach, mocker):
    utils.check_val_logging(approach, mocker)


@torch.no_grad()
def test_test_step_logging(approach, mocker):
    utils.check_test_logging(approach, mocker)


def test_model_hparams_logged(models, mocker):
    approach = DannApproach(1.0, lr=0.001)
    mocker.patch.object(approach, "log_model_hyperparameters")

    approach.set_model(*models)

    approach.log_model_hyperparameters.assert_called_with("domain_disc")


def test_checkpointing(tmp_path):
    ckpt_path = tmp_path / "checkpoint.ckpt"
    fe = model.CnnExtractor(1, [16], 10, fc_units=16)
    reg = model.FullyConnectedHead(16, [1])
    disc = model.FullyConnectedHead(16, [1], act_func_on_last_layer=False)
    approach = DannApproach(1.0, lr=0.01)
    approach.set_model(fe, reg, disc)

    utils.checkpoint(approach, ckpt_path)
    DannApproach.load_from_checkpoint(ckpt_path)


@pytest.mark.integration
def test_on_dummy():
    pl.seed_everything(42)

    fd3 = rul_datasets.reader.DummyReader(fd=1)
    fd1 = fd3.get_compatible(fd=2, percent_broken=0.8)
    dm = rul_datasets.DomainAdaptionDataModule(
        rul_datasets.RulDataModule(fd3, 16), rul_datasets.RulDataModule(fd1, 16)
    )

    feature_extractor = rul_adapt.model.LstmExtractor(1, [16], 8)
    regressor = rul_adapt.model.FullyConnectedHead(8, [1], act_func_on_last_layer=False)
    disc = rul_adapt.model.FullyConnectedHead(
        8, [8, 8, 1], act_func_on_last_layer=False
    )

    approach = rul_adapt.approach.DannApproach(4.0, lr=0.01, loss_type="mae")
    approach.set_model(feature_extractor, regressor, disc)

    trainer = pl.Trainer(
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        max_epochs=10,
        gradient_clip_val=1.0,
    )
    trainer.fit(approach, dm)
    trainer.test(approach, dm)
