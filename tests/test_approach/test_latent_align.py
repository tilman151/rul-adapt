from unittest import mock

import numpy as np
import numpy.testing as npt
import pytest
import rul_datasets.reader
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Metric
import pytorch_lightning as pl

from rul_adapt import model
from rul_adapt.approach.latent_align import (
    LatentAlignApproach,
    extract_chunk_windows,
    get_first_time_to_predict,
    LatentAlignFttpApproach,
    get_health_indicator,
    ChunkWindowExtractor,
)
from tests.test_approach import utils


@pytest.fixture()
def models():
    fe = model.CnnExtractor(14, [8], 30, fc_units=4)
    reg = model.FullyConnectedHead(4, [1], act_func_on_last_layer=False)

    return fe, reg


@pytest.fixture()
def approach(models):
    feature_extractor, regressor = models
    approach = LatentAlignApproach(1.0, 1.0, 1.0, 1.0, lr=0.001)
    approach.set_model(feature_extractor, regressor)

    return approach


@pytest.fixture()
def inputs():
    return (
        torch.randn(10, 14, 30),  # source
        torch.arange(10, dtype=torch.float),  # source_degradation_steps
        torch.arange(10, dtype=torch.float),  # source_labels
        torch.randn(10, 14, 30),  # target
        torch.arange(10, dtype=torch.float),  # target_degradation_steps
        torch.randn(10, 14, 30),  # healthy
    )


class TestLatentAlignApproach:
    def test_optimizer_configured_with_factory(self, models, mocker):
        mock_factory = mocker.patch("rul_adapt.utils.OptimizerFactory")
        kwargs = {"optim_type": "sgd", "lr": 0.001, "optim_weight_decay": 0.001}
        approach = LatentAlignApproach(1.0, 1.0, 1.0, 1.0, **kwargs)
        approach.configure_optimizers()

        mock_factory.assert_called_once_with(**kwargs)
        mock_factory().assert_called_once()

    @pytest.mark.parametrize("as_percentage", [True, False])
    def test_training_labels_as_percentage(
        self, approach, inputs, as_percentage, mocker
    ):
        mock_as_percentage = mocker.patch.object(
            approach, "_to_percentage", return_value=inputs[2][:, None]
        )
        approach.labels_as_percentage = as_percentage

        approach.training_step(inputs, 0)

        if as_percentage:
            mock_as_percentage.assert_called_once()
        else:
            mock_as_percentage.assert_not_called()

    @pytest.mark.parametrize("as_percentage", [True, False])
    def test_forward_labels_as_percentage(
        self, approach, inputs, as_percentage, mocker
    ):
        mock_from_percentage = mocker.patch.object(approach, "_from_percentage")
        approach.labels_as_percentage = as_percentage
        features, *_ = inputs

        approach(features)

        if as_percentage:
            mock_from_percentage.assert_called_once()
        else:
            mock_from_percentage.assert_not_called()

    def test_forward(self, approach, inputs):
        features, *_ = inputs

        outputs = approach(features)

        assert outputs.shape == torch.Size([10, 1])

    def test_train_step(self, approach, inputs):

        outputs = approach.training_step(inputs, 0)

        assert outputs.shape == torch.Size([])

    def test_val_step(self, approach, inputs):
        features, labels, *_ = inputs

        approach.validation_step([features, labels], 0, dataloader_idx=0)
        approach.validation_step([features, labels], 0, dataloader_idx=1)
        with pytest.raises(RuntimeError):
            approach.validation_step([features, labels], 0, dataloader_idx=2)

    def test_test_step(self, approach, inputs):
        features, labels, *_ = inputs

        approach.test_step([features, labels], 0, dataloader_idx=0)
        approach.test_step([features, labels], 0, dataloader_idx=1)
        with pytest.raises(RuntimeError):
            approach.test_step([features, labels], 0, dataloader_idx=2)

    def test_train_step_logging(self, approach, inputs):
        approach.train_mse = mock.MagicMock(Metric)
        approach.healthy_align = mock.MagicMock(Metric)
        approach.direction_align = mock.MagicMock(Metric)
        approach.level_align = mock.MagicMock(Metric)
        approach.fusion_align = mock.MagicMock(nn.Module, return_value=torch.zeros(1))
        approach.log = mock.MagicMock()

        approach.training_step(inputs, 0)

        approach.train_mse.assert_called_once()
        approach.healthy_align.assert_called_once()
        approach.direction_align.assert_called_once()
        approach.level_align.assert_called_once()
        approach.fusion_align.assert_called_once()

        approach.log.assert_has_calls(
            [
                mock.call("train/loss", mock.ANY),
                mock.call("train/mse", approach.train_mse),
                mock.call("train/healthy_align", approach.healthy_align),
                mock.call("train/direction_align", approach.direction_align),
                mock.call("train/level_align", approach.level_align),
                mock.call("train/fusion_align", approach.fusion_align),
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
        approach = LatentAlignApproach(0.1, 0.1, 0.1, 0.1, lr=0.001)
        approach.set_model(fe, reg)

        utils.checkpoint(approach, ckpt_path)
        LatentAlignApproach.load_from_checkpoint(ckpt_path)

    @pytest.mark.integration
    def test_on_dummy(self):
        source = rul_datasets.reader.DummyReader(fd=1)
        target = source.get_compatible(fd=2, percent_broken=0.8)
        dm = rul_datasets.LatentAlignDataModule(
            rul_datasets.RulDataModule(source, 32),
            rul_datasets.RulDataModule(target, 32),
            split_by_steps=20,
        )

        fe = model.CnnExtractor(1, [16, 16], 10, fc_units=16)
        reg = model.FullyConnectedHead(16, [1], act_func_on_last_layer=False)
        approach = LatentAlignApproach(0.1, 0.1, 0.1, 0.1, lr=0.001)
        approach.set_model(fe, reg)

        trainer = pl.Trainer(
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
            max_epochs=10,
        )
        trainer.fit(approach, dm)
        trainer.test(approach, dm)


class TestLatentAlignFttpApproach:
    @pytest.mark.parametrize(
        "gen",
        [
            model.CnnExtractor(1, [16], 128),
            pytest.param(None, marks=pytest.mark.xfail(strict=True)),
        ],
    )
    def test_set_model(self, models, gen):
        fe, reg = models
        fttp_approach = LatentAlignFttpApproach(128, lr=0.001)

        fttp_approach.set_model(fe, reg, gen)

        assert gen == fttp_approach.generator

    def test_training_step(self, models):
        fe, reg = models
        gen = nn.Conv1d(1, 14, 10, padding="same")
        fttp_approach = LatentAlignFttpApproach(30, lr=0.001)
        fttp_approach.set_model(fe, reg, gen)

        inputs = (torch.randn(10, 14, 30), torch.zeros(10))
        loss = fttp_approach.training_step(inputs)

        assert loss.shape == torch.Size([])

    @pytest.mark.integration
    def test_on_dummy(self):
        dummy = rul_datasets.reader.DummyReader(1)
        healthy, _ = rul_datasets.adaption.split_healthy(
            *dummy.load_split("dev"), by_steps=10
        )
        healthy_dl = DataLoader(healthy, batch_size=32, shuffle=True)

        fe = model.CnnExtractor(1, [16, 16], 10, fc_units=16)
        reg = model.FullyConnectedHead(16, [1], act_func_on_last_layer=False)
        gen = model.CnnExtractor(1, [1], 10, 3, padding=True)
        fttp_approach = LatentAlignFttpApproach(10, lr=0.001)
        fttp_approach.set_model(fe, reg, gen)

        trainer = pl.Trainer(
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
            max_epochs=10,
        )
        trainer.fit(fttp_approach, healthy_dl)


@pytest.mark.parametrize(
    ["window_size", "chunk_size"], [(7, 4), (2, 3), [5, 12], [30, 1]]
)
def test_extract_chunk_windows(window_size, chunk_size):
    num_windows = 30
    old_window_size = 12
    data = np.arange(num_windows * old_window_size).reshape(
        (num_windows, old_window_size, 1)
    )
    windows = extract_chunk_windows(data, window_size, chunk_size)

    # windows span whole data
    assert windows[0, 0, 0] == 0
    assert windows[-1, -1, 0] == num_windows * old_window_size - 1

    # chunks are extracted correctly
    num_window_groups = len(windows) // (old_window_size // chunk_size)
    for g, window_group in enumerate(np.split(windows, num_window_groups)):
        # each window group contains windows that start at the same old window
        for w, window in enumerate(window_group):
            # each window contains one chunk of each consecutive window
            for c, chunk in enumerate(np.split(window, window_size)):
                exp_start_idx = (
                    g * old_window_size + w * chunk_size + c * old_window_size
                )
                npt.assert_equal(
                    chunk.squeeze(),
                    np.arange(exp_start_idx, exp_start_idx + chunk_size),
                )


def test_chunk_window_extractor():
    window_size = 5
    extractor = ChunkWindowExtractor(window_size, 2)
    features, targets = np.arange(100).reshape((10, 10, 1)), np.arange(10)

    chunked_features, chunked_targets = extractor(features, targets)

    assert len(chunked_features) == len(chunked_targets)
    assert chunked_targets[0] == window_size - 1


@mock.patch("rul_adapt.approach.latent_align.get_health_indicator")
def test_get_first_time_to_predict(mock_get_health_indicator):
    window_size = 5
    health_index = (
        np.maximum(0, 10 * np.linspace(-5, 5, 100)) + 1 + np.random.randn(100) * 0.001
    )
    mock_get_health_indicator.return_value = health_index
    fttp_model = mock.MagicMock(nn.Module)
    data = np.random.randn(104, 10, 1)

    fttp = get_first_time_to_predict(
        fttp_model,
        data,
        window_size,
        chunk_size=2,
        healthy_index=10,
        threshold_coefficient=1.5,
    )

    offset = len(data) - len(health_index)
    assert fttp == (50 + offset)  # fttp index is adjusted for window size
    mock_get_health_indicator.assert_called_once_with(fttp_model, data, window_size, 2)


@mock.patch("rul_adapt.approach.latent_align.extract_chunk_windows")
def test_get_health_indicator(mock_extract_chunk_windows):
    chunk_windows = np.random.randn(100, 1280, 1)
    mock_extract_chunk_windows.return_value = chunk_windows

    model = mock.MagicMock(nn.Module)
    model.side_effect = torch.stack([torch.arange(0, 20, 2), torch.zeros(10)], dim=1)
    features = np.random.randn(19, 1280, 1)

    health_index = get_health_indicator(model, features, 10, 128)

    assert health_index.shape == (10,)
    npt.assert_equal(health_index, np.arange(10))  # std of model outputs is range(10)
    mock_extract_chunk_windows.assert_called_once_with(features, 10, 128)
    model.assert_called()
