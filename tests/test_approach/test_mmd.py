from unittest import mock

import pytest
import torch
import torchmetrics
from torch import nn
from torchmetrics import Metric

import rul_adapt.loss
from rul_adapt import model
from rul_adapt.approach.mmd import MmdApproach
from tests.test_approach import utils


@pytest.fixture()
def models():
    fe = model.LstmExtractor(14, [16], 4)
    reg = model.FullyConnectedHead(4, [1], act_func_on_last_layer=False)

    return fe, reg


@pytest.fixture()
def approach(models):
    feature_extractor, regressor = models
    approach = MmdApproach(0.1, lr=0.001)
    approach.set_model(feature_extractor, regressor)

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
    kwargs = {"optim_type": "sgd", "lr": 0.001, "optim_weight_decay": 0.001}
    approach = MmdApproach(0.01, **kwargs)
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
    approach = MmdApproach(0.001, loss_type=loss_type, lr=0.001)

    assert approach.loss_type == loss_type
    assert approach.train_source_loss == expected


def test_num_mmd_kernels():
    approach = MmdApproach(0.1, num_mmd_kernels=3, lr=0.001)

    assert approach.mmd_loss.num_kernels == 3


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
    approach.mmd_loss = mock.MagicMock(rul_adapt.loss.MaximumMeanDiscrepancyLoss)
    approach.log = mock.MagicMock()

    approach.training_step(inputs, batch_idx=0)

    approach.train_source_loss.assert_called_once()
    approach.mmd_loss.assert_called_once()

    approach.log.assert_has_calls(
        [
            mock.call("train/loss", mock.ANY),
            mock.call("train/source_loss", approach.train_source_loss),
            mock.call("train/mmd", approach.mmd_loss),
        ]
    )


@torch.no_grad()
def test_val_step_logging(approach, mocker):
    utils.check_val_logging(approach, mocker)


@torch.no_grad()
def test_test_step_logging(approach, mocker):
    utils.check_test_logging(approach, mocker)


def test_checkpointing(tmp_path):
    ckpt_path = tmp_path / "checkpoint.ckpt"
    fe = model.CnnExtractor(1, [16], 10, fc_units=16)
    reg = model.FullyConnectedHead(16, [1])
    approach = MmdApproach(0.1, lr=0.001)
    approach.set_model(fe, reg)

    utils.checkpoint(approach, ckpt_path)
    MmdApproach.load_from_checkpoint(ckpt_path)
