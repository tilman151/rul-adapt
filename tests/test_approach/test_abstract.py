from unittest import mock

import pytest
import pytorch_lightning as pl
import rul_datasets
import torch.optim

from rul_adapt import model
from rul_adapt.approach.abstract import AdaptionApproach


class DummyApproach(AdaptionApproach):
    def __init__(self):
        super().__init__()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.001)

    def training_step(self, *args, **kwargs):
        pass


def test_set_model():
    mock_feature_extractor = mock.MagicMock()
    mock_regressor = mock.MagicMock()
    approach = DummyApproach()

    approach.set_model(mock_feature_extractor, mock_regressor)

    assert approach.feature_extractor is mock_feature_extractor
    assert approach.regressor is mock_regressor


def test_feature_extractor():
    approach = DummyApproach()

    with pytest.raises(RuntimeError):
        _ = approach.feature_extractor


def test_regressor():
    approach = DummyApproach()

    with pytest.raises(RuntimeError):
        _ = approach.regressor


def test_checkpointing(tmp_path):
    ckpt_path = tmp_path / "checkpoint.ckpt"

    fe = model.CnnExtractor(14, [8], 10, fc_units=16)
    reg = model.FullyConnectedHead(16, [1])
    approach = DummyApproach()
    approach.set_model(fe, reg)
    dm = rul_datasets.RulDataModule(rul_datasets.reader.DummyReader(1), 32)

    trainer = pl.Trainer(max_epochs=0)
    trainer.fit(approach, dm)
    trainer.save_checkpoint(ckpt_path)
    restored = DummyApproach.load_from_checkpoint(ckpt_path)

    paired_params = zip(approach.parameters(), restored.parameters())
    for org_weight, restored_weight in paired_params:
        assert torch.dist(org_weight, restored_weight) == 0.0
        assert org_weight.requires_grad == restored_weight.requires_grad


@pytest.mark.integration
def test_dummy_integration():
    """This is a dummy test so that the integration workflow does not report missing
    test cases when no approach has integration tests yet."""
    pass
