from unittest import mock

import pytorch_lightning as pl
import rul_datasets
import torch


def checkpoint(approach, ckpt_path, max_rul=50):
    dm = rul_datasets.RulDataModule(
        rul_datasets.reader.DummyReader(1, max_rul=max_rul), 32
    )
    trainer = pl.Trainer(max_epochs=0, num_sanity_val_steps=0, logger=False)
    trainer.fit(approach, dm)
    trainer.save_checkpoint(ckpt_path)


def check_val_logging(approach, mocker):
    mocker.patch.object(approach, "evaluator", autospec=True)
    mocker.patch.object(approach, "feature_extractor", autospec=True)
    mocker.patch.object(approach, "regressor", autospec=True)

    features, labels = mock.MagicMock(torch.Tensor), mock.MagicMock(torch.Tensor)
    # check source data loader call
    approach.validation_step([features, labels], batch_idx=0, dataloader_idx=0)
    approach.evaluator.validation.assert_called_with([features, labels], "source")
    approach.evaluator.reset_mock()
    # check target data loader call
    approach.validation_step([features, labels], batch_idx=0, dataloader_idx=1)
    approach.evaluator.validation.assert_called_with([features, labels], "target")


def check_test_logging(approach, mocker):
    mocker.patch.object(approach, "evaluator", autospec=True)
    mocker.patch.object(approach, "feature_extractor", autospec=True)
    mocker.patch.object(approach, "regressor", autospec=True)

    features, labels = mock.MagicMock(), mock.MagicMock()
    # check source data loader call
    approach.test_step([features, labels], batch_idx=0, dataloader_idx=0)
    approach.evaluator.test.assert_called_with([features, labels], "source")
    approach.evaluator.reset_mock()
    # check target data loader call
    approach.test_step([features, labels], batch_idx=0, dataloader_idx=1)
    approach.evaluator.test.assert_called_with([features, labels], "target")
