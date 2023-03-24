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
    approach = LatentAlignApproach(1.0, 1.0, 1.0, 1.0, 0.001)
    approach.set_model(feature_extractor, regressor)

    return approach


@pytest.fixture()
def mocked_approach():
    feature_extractor = mock.MagicMock(nn.Module, return_value=torch.zeros(10, 8))
    regressor = mock.MagicMock(nn.Module, return_value=torch.zeros(10, 1))
    approach = LatentAlignApproach(1.0, 1.0, 1.0, 1.0, 0.001)
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

    def test_train_step_logging(self, mocked_approach, inputs):
        approach = mocked_approach
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
    def test_val_step_logging(self, mocked_approach, inputs):
        approach = mocked_approach
        features, labels, *_ = inputs
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
    def test_test_step_logging(self, mocked_approach, inputs):
        approach = mocked_approach
        features, labels, *_ = inputs
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

    def test_checkpointing(self, tmp_path):
        ckpt_path = tmp_path / "checkpoint.ckpt"
        fe = model.CnnExtractor(1, [16], 10, fc_units=16)
        reg = model.FullyConnectedHead(16, [1])
        approach = LatentAlignApproach(0.1, 0.1, 0.1, 0.1, 0.001)
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
        approach = LatentAlignApproach(0.1, 0.1, 0.1, 0.1, 0.001)
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
        fttp_approach = LatentAlignFttpApproach(0.01, 128)

        fttp_approach.set_model(fe, reg, gen)

        assert gen == fttp_approach.generator

    def test_training_step(self, models):
        fe, reg = models
        gen = nn.Conv1d(1, 14, 10, padding="same")
        fttp_approach = LatentAlignFttpApproach(0.01, 30)
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
        healthy_dl = DataLoader(healthy, batch_size=128, shuffle=True)

        fe = model.CnnExtractor(1, [16, 16], 10, fc_units=16)
        reg = model.FullyConnectedHead(16, [1], act_func_on_last_layer=False)
        gen = nn.Conv1d(1, 1, 10, padding="same")
        fttp_approach = LatentAlignFttpApproach(0.01, 10)
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


def test_get_first_time_to_predict():
    window_size = 5
    health_index = (
        np.maximum(0, 10 * np.linspace(-5, 5, 500)) + 1 + np.random.randn(500) * 0.001
    )
    data = np.random.randn(100 + window_size - 1, 10, 1)
    fttp_model = mock.Mock(
        side_effect=torch.from_numpy(health_index.reshape(100, -1, 1))
    )
    fttp = get_first_time_to_predict(
        fttp_model,
        data,
        window_size,
        chunk_size=2,
        healthy_index=10,
        threshold_coefficient=10,
    )

    assert fttp == (50 + window_size - 1)
