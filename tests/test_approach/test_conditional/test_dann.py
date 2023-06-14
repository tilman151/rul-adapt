from unittest import mock

import pytest
import pytorch_lightning as pl
import rul_datasets.reader
import torch
import torchmetrics
from torchmetrics import Metric

from rul_adapt import model, loss
from rul_adapt.approach.conditional import ConditionalDannApproach
from tests.test_approach import utils


@pytest.fixture()
def models():
    fe = model.LstmExtractor(14, [16], 4)
    reg = model.FullyConnectedHead(4, [1], act_func_on_last_layer=False)
    domain_disc = model.FullyConnectedHead(4, [1], act_func_on_last_layer=False)

    return fe, reg, domain_disc


@pytest.fixture()
def approach(models):
    feature_extractor, regressor, domain_disc = models
    approach = ConditionalDannApproach(0.1, 0.5, [(0.0, 1.0)], lr=0.001)
    approach.set_model(feature_extractor, regressor, domain_disc)

    return approach


@pytest.fixture()
def inputs():
    return (
        torch.randn(10, 14, 30),  # source
        torch.arange(10, dtype=torch.float),  # source_labels
        torch.randn(10, 14, 30),  # target
    )


def test_optimizer_configured_with_factory(models, mocker):
    mock_factory = mocker.patch("rul_adapt.utils.OptimizerFactory")
    kwargs = {"optim_type": "sgd", "lr": 0.001, "weight_decay": 0.001}
    approach = ConditionalDannApproach(0.1, 0.5, [(0.0, 1.0)], **kwargs)
    approach.set_model(*models)
    approach.configure_optimizers()

    mock_factory.assert_called_once_with(**kwargs)
    mock_factory().assert_called_once()


@pytest.mark.parametrize(
    ["loss_type", "expected"],
    [
        ("mae", torchmetrics.MeanAbsoluteError()),
        ("mse", torchmetrics.MeanSquaredError()),
        ("rmse", torchmetrics.MeanSquaredError(squared=False)),
    ],
)
def test_loss_type(loss_type, expected):
    approach = ConditionalDannApproach(
        0.001, 0.5, [(0.0, 1.0)], loss_type=loss_type, lr=0.001
    )

    assert approach.loss_type == loss_type
    assert approach.train_source_loss == expected


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
    approach.dann_loss = mock.MagicMock(loss.MaximumMeanDiscrepancyLoss)
    approach.conditional_dann_loss = mock.MagicMock(loss.ConditionalAdaptionLoss)
    approach.log = mock.MagicMock()

    approach.training_step(inputs, batch_idx=0)

    approach.train_source_loss.assert_called_once()
    approach.dann_loss.assert_called_once()

    approach.log.assert_has_calls(
        [
            mock.call("train/loss", mock.ANY),
            mock.call("train/source_loss", approach.train_source_loss),
            mock.call("train/dann", approach.dann_loss),
            mock.call("train/conditional_dann", approach.conditional_dann_loss),
        ]
    )


@torch.no_grad()
def test_val_step_logging(approach, mocker):
    utils.check_val_logging(approach, mocker)


@torch.no_grad()
def test_test_step_logging(approach, mocker):
    utils.check_test_logging(approach, mocker)


def test_model_hparams_logged(models, mocker):
    approach = ConditionalDannApproach(1.0, 0.5, [(0.0, 1.0)])
    mocker.patch.object(approach, "log_model_hyperparameters")

    approach.set_model(*models)

    approach.log_model_hyperparameters.assert_called_with("domain_disc")


def test_checkpointing(tmp_path, approach):
    ckpt_path = tmp_path / "checkpoint.ckpt"

    utils.checkpoint(approach, ckpt_path)
    ConditionalDannApproach.load_from_checkpoint(ckpt_path)


@pytest.mark.integration
def test_on_dummy():
    source = rul_datasets.reader.DummyReader(fd=1)
    target = source.get_compatible(fd=2, percent_broken=0.8)
    dm = rul_datasets.DomainAdaptionDataModule(
        rul_datasets.RulDataModule(source, 32), rul_datasets.RulDataModule(target, 32)
    )

    fe = model.CnnExtractor(1, [16, 16], 10, fc_units=16)
    reg = model.FullyConnectedHead(16, [1], act_func_on_last_layer=False)
    domain_disc = model.FullyConnectedHead(16, [1], act_func_on_last_layer=False)
    approach = ConditionalDannApproach(
        1.0, 0.5, [(50, 30), (40, 20), (30, 0)], lr=0.001
    )
    approach.set_model(fe, reg, domain_disc)

    trainer = pl.Trainer(
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        max_epochs=10,
    )
    trainer.fit(approach, dm)
    trainer.test(approach, dm)
