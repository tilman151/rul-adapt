from unittest import mock

import pytest
import torch
from torch import nn
from torchmetrics import Metric

from rul_adapt import model
from rul_adapt.approach.latent_align import LatentAlignApproach


@pytest.fixture()
def models():
    fe = model.CnnExtractor(14, [8], 30, fc_units=4)
    reg = model.FullyConnectedHead(4, [1], act_func_on_last_layer=False)

    return fe, reg


@pytest.fixture()
def approach(models):
    feature_extractor, regressor = models
    approach = LatentAlignApproach(1.0, 1.0, 1.0, 1.0, 125, 0.001)
    approach.set_model(feature_extractor, regressor)

    return approach


@pytest.fixture()
def mocked_approach():
    feature_extractor = mock.MagicMock(nn.Module, return_value=torch.zeros(10, 8))
    regressor = mock.MagicMock(nn.Module, return_value=torch.zeros(10, 1))
    approach = LatentAlignApproach(1.0, 1.0, 1.0, 1.0, 125, 0.001)
    approach.set_model(feature_extractor, regressor)

    return approach


@pytest.fixture()
def inputs():
    return (
        torch.randn(10, 14, 30),  # healthy
        torch.randn(10, 14, 30),  # source
        torch.arange(10, dtype=torch.float)[:, None],  # source_labels
        torch.randn(10, 14, 30),  # target
        torch.arange(10, dtype=torch.float)[:, None],  # target_degradation_steps
    )


def test_forward(approach, inputs):
    _, features, _, _, _ = inputs

    outputs = approach(features)

    assert outputs.shape == torch.Size([10, 1])


def test_train_step(approach, inputs):

    outputs = approach.training_step(*inputs)

    assert outputs.shape == torch.Size([])


def test_val_step(approach, inputs):
    _, features, labels, _, _ = inputs

    approach.validation_step(features, labels, dataloader_idx=0)
    approach.validation_step(features, labels, dataloader_idx=1)
    with pytest.raises(RuntimeError):
        approach.validation_step(features, labels, dataloader_idx=2)


def test_test_step(approach, inputs):
    _, features, labels, _, _ = inputs

    approach.test_step(features, labels, dataloader_idx=0)
    approach.test_step(features, labels, dataloader_idx=1)
    with pytest.raises(RuntimeError):
        approach.test_step(features, labels, dataloader_idx=2)


def test_train_step_logging(mocked_approach, inputs):
    approach = mocked_approach
    approach.train_mse = mock.MagicMock(Metric)
    approach.healthy_align = mock.MagicMock(Metric)
    approach.direction_align = mock.MagicMock(Metric)
    approach.level_align = mock.MagicMock(Metric)
    approach.fusion_align = mock.MagicMock(nn.Module, return_value=torch.zeros(1))
    approach.log = mock.MagicMock()

    approach.training_step(*inputs)

    approach.train_mse.assert_called_once()
    approach.healthy_align.assert_called_once()
    approach.direction_align.assert_called_once()
    approach.level_align.assert_called_once()
    approach.fusion_align.assert_called_once()

    approach.log.assert_has_calls(
        [
            mock.call("loss", mock.ANY),
            mock.call("mse", approach.train_mse),
            mock.call("healthy_align", approach.healthy_align),
            mock.call("direction_align", approach.direction_align),
            mock.call("level_align", approach.level_align),
            mock.call("fusion_align", torch.zeros(1)),
        ]
    )


def test_val_step_logging(mocked_approach, inputs):
    approach = mocked_approach
    _, features, labels, _, _ = inputs
    approach.val_source_mse = mock.MagicMock(Metric)
    approach.val_target_mse = mock.MagicMock(Metric)
    approach.log = mock.MagicMock()

    # check source data loader call
    approach.validation_step(features, labels, dataloader_idx=0)
    approach.val_source_mse.assert_called_once()
    approach.log.assert_called_with("val_source_mse", approach.val_source_mse)
    approach.val_target_mse.assert_not_called()

    approach.val_source_mse.reset_mock()

    # check target data loader call
    approach.validation_step(features, labels, dataloader_idx=1)
    approach.val_source_mse.assert_not_called()
    approach.val_target_mse.assert_called_once()
    approach.log.assert_called_with("val_target_mse", approach.val_target_mse)


def test_test_step_logging(mocked_approach, inputs):
    approach = mocked_approach
    _, features, labels, _, _ = inputs
    approach.test_source_mse = mock.MagicMock(Metric)
    approach.test_target_mse = mock.MagicMock(Metric)
    approach.log = mock.MagicMock()

    # check source data loader call
    approach.test_step(features, labels, dataloader_idx=0)
    approach.test_source_mse.assert_called_once()
    approach.log.assert_called_with("test_source_mse", approach.test_source_mse)
    approach.test_target_mse.assert_not_called()

    approach.test_source_mse.reset_mock()

    # check target data loader call
    approach.test_step(features, labels, dataloader_idx=1)
    approach.test_source_mse.assert_not_called()
    approach.test_target_mse.assert_called_once()
    approach.log.assert_called_with("test_target_mse", approach.test_target_mse)
