from unittest import mock

import pytest
import pytorch_lightning as pl
import rul_datasets
import torch.optim
from torch import nn

from rul_adapt import model
from rul_adapt.approach.abstract import AdaptionApproach


class DummyApproach(AdaptionApproach):
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.001)

    def training_step(self, *args, **kwargs):
        pass


class DummyApproachWithExtraModel(DummyApproach):
    CHECKPOINT_MODELS = ["_extra"]

    _extra: nn.Module

    def set_model(
        self, feature_extractor, regressor, extra=None, *args, **kwargs
    ) -> None:
        super().set_model(feature_extractor, regressor)
        self._extra = extra


class UncheckpointableModel(nn.Module):
    """To be checkpoint-able with this library's mechanism, a model has to save all
    arguments of its constructor as member variables under the same name. Otherwise,
    the arguments the model was initialized with cannot be retrieved."""

    def __init__(self, a):
        super().__init__()

        self.b = a  # init arg is not saved under same name as a member var
        self.layer = nn.Linear(self.b, 1)

    def forward(self, inputs):
        return self.layer(inputs)


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


@pytest.mark.parametrize(
    ["approach", "models"],
    [
        (
            DummyApproach(),
            [
                model.CnnExtractor(14, [8], 10, fc_units=16),
                model.FullyConnectedHead(16, [1]),
            ],
        ),
        (
            DummyApproach(),
            [
                model.LstmExtractor(14, [16], 16),
                model.FullyConnectedHead(16, [1]),
            ],
        ),
        (
            DummyApproachWithExtraModel(),
            [
                model.CnnExtractor(14, [8], 10, fc_units=16),
                model.FullyConnectedHead(16, [1]),
                model.FullyConnectedHead(16, [10, 1]),
            ],
        ),
    ],
)
def test_checkpointing(tmp_path, approach, models):
    ckpt_path = tmp_path / "checkpoint.ckpt"
    approach.set_model(*models)
    _checkpoint(approach, ckpt_path)
    restored = type(approach).load_from_checkpoint(ckpt_path)

    paired_params = zip(approach.parameters(), restored.parameters())
    for org_weight, restored_weight in paired_params:
        assert torch.dist(org_weight, restored_weight) == 0.0
        assert org_weight.requires_grad == restored_weight.requires_grad


def test_error_on_uncheckpointable_model(tmp_path):
    fe = model.LstmExtractor(14, [16], 16)
    reg = UncheckpointableModel(16)
    approach = DummyApproach()
    approach.set_model(fe, reg)

    with pytest.raises(RuntimeError) as exc_info:
        _checkpoint(approach, tmp_path / "checkpoint.ckpt")
        assert exc_info.value.args[0].startswith("The object of type ")


def _checkpoint(approach, ckpt_path):
    dm = rul_datasets.RulDataModule(rul_datasets.reader.DummyReader(1), 32)
    trainer = pl.Trainer(max_epochs=0)
    trainer.fit(approach, dm)
    trainer.save_checkpoint(ckpt_path)


@pytest.mark.integration
def test_dummy_integration():
    """This is a dummy test so that the integration workflow does not report missing
    test cases when no approach has integration tests yet."""
    pass
