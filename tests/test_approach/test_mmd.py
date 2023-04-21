from unittest import mock

import pytest
import rul_datasets.reader
import torch
import torchmetrics
import pytorch_lightning as pl
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
    approach = MmdApproach(0.001, 0.1)
    approach.set_model(feature_extractor, regressor)

    return approach


@pytest.fixture()
def mocked_approach():
    feature_extractor = mock.MagicMock(nn.Module, return_value=torch.zeros(10, 8))
    regressor = mock.MagicMock(nn.Module, return_value=torch.zeros(10, 1))
    approach = MmdApproach(0.001, 0.1)
    approach.set_model(feature_extractor, regressor)

    return approach


@pytest.fixture()
def inputs():
    return (
        torch.randn(10, 14, 30),  # source
        torch.arange(10, dtype=torch.float),  # source_labels
        torch.randn(10, 14, 30),  # target
    )


def test_configure_optimizer(models):
    approach = MmdApproach(0.001, 0.1)
    approach.set_model(*models)

    optim = approach.configure_optimizers()

    assert isinstance(optim, torch.optim.Adam)
    assert optim.defaults["lr"] == 0.001
    assert list(approach.parameters()) == optim.param_groups[0]["params"]


@pytest.mark.parametrize(
    ["loss_type", "expected"],
    [
        ("mae", torchmetrics.MeanAbsoluteError()),
        ("mse", torchmetrics.MeanSquaredError()),
        ("rmse", torchmetrics.MeanSquaredError(squared=False)),
    ],
)
def test_loss_type(loss_type, expected):
    approach = MmdApproach(1.0, 0.001, loss_type=loss_type)

    assert approach.loss_type == loss_type
    assert approach.train_source_loss == expected
    assert not approach.val_source_rmse.squared  # val and test always uses rmse
    assert not approach.val_target_rmse.squared
    assert not approach.test_source_rmse.squared
    assert not approach.test_target_rmse.squared


def test_num_mmd_kernels():
    approach = MmdApproach(0.01, 0.1, num_mmd_kernels=3)

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
def test_train_step_logging(mocked_approach, inputs):
    approach = mocked_approach
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
def test_val_step_logging(mocked_approach, inputs):
    approach = mocked_approach
    features, labels, _ = inputs
    approach.val_source_rmse = mock.MagicMock(Metric)
    approach.val_source_score = mock.MagicMock(Metric)
    approach.val_target_rmse = mock.MagicMock(Metric)
    approach.val_target_score = mock.MagicMock(Metric)
    approach.log = mock.MagicMock()

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
def test_test_step_logging(mocked_approach, inputs):
    approach = mocked_approach
    features, labels, _ = inputs
    approach.test_source_rmse = mock.MagicMock(Metric)
    approach.test_source_score = mock.MagicMock(Metric)
    approach.test_target_rmse = mock.MagicMock(Metric)
    approach.test_target_score = mock.MagicMock(Metric)
    approach.log = mock.MagicMock()

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


def test_checkpointing(tmp_path):
    ckpt_path = tmp_path / "checkpoint.ckpt"
    fe = model.CnnExtractor(1, [16], 10, fc_units=16)
    reg = model.FullyConnectedHead(16, [1])
    approach = MmdApproach(0.001, 0.1)
    approach.set_model(fe, reg)

    utils.checkpoint(approach, ckpt_path)
    MmdApproach.load_from_checkpoint(ckpt_path)
