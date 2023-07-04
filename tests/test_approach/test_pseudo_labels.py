from unittest import mock

import numpy as np
import pytest
import rul_datasets
import torch

from rul_adapt import model
from rul_adapt.approach import SupervisedApproach
from rul_adapt.approach.pseudo_labels import (
    generate_pseudo_labels,
    patch_pseudo_labels,
    _PseudoLabelReader,
)


@pytest.mark.parametrize(["inductive", "exp_split"], [(True, "test"), (False, "dev")])
def test_generate_pseudo_labels(mocker, inductive, exp_split):
    fe = model.CnnExtractor(1, [3], 10, fc_units=16)
    reg = model.FullyConnectedHead(16, [1], act_func_on_last_layer=False)
    approach = SupervisedApproach("mse")
    approach.set_model(fe, reg)
    mock_dm = mocker.MagicMock(rul_datasets.RulDataModule)
    mock_dm.reader.max_rul = 50
    mock_dm.load_split.return_value = ([torch.randn(15, 1, 10)], [torch.randn(15)])

    pseudo_labels = generate_pseudo_labels(mock_dm, approach, inductive=inductive)

    assert len(pseudo_labels) == 1
    assert all([isinstance(pl, float) for pl in pseudo_labels])
    mock_dm.load_split.assert_called_once_with(exp_split, alias="dev")


def test_generate_pseudo_labels_max_rul_warning():
    approach = mock.MagicMock(SupervisedApproach)
    approach.return_value = torch.arange(45.0, 55.0)[:, None]  # default max_rul is 50
    dm = rul_datasets.RulDataModule(rul_datasets.reader.DummyReader(1), 32)

    with pytest.warns(UserWarning):
        generate_pseudo_labels(dm, approach)


def test_generate_pseudo_labels_negative_rul_warning():
    approach = mock.MagicMock(SupervisedApproach)
    approach.return_value = torch.arange(-1.0, 9.0)[:, None]
    dm = rul_datasets.RulDataModule(rul_datasets.reader.DummyReader(1), 32)

    with pytest.warns(UserWarning):
        generate_pseudo_labels(dm, approach)


@pytest.mark.parametrize("inductive", [True, False])
def test_patch_pseudo_labels(inductive):
    dm = rul_datasets.RulDataModule(rul_datasets.reader.DummyReader(1), 32)
    dm.setup = mock.Mock(wraps=dm.setup)
    pseudo_labels = [float(i) for i in range(40, 50)]

    patch_pseudo_labels(dm, pseudo_labels, inductive=inductive)

    assert isinstance(dm.reader, _PseudoLabelReader)
    assert dm.reader._inductive == inductive
    dm.setup.assert_called()


class TestPseudoLabelReader:
    def test_max_rul_error(self):
        reader = rul_datasets.reader.DummyReader(1)
        pseudo_labels = [float(i) for i in range(45, 55)]

        with pytest.raises(ValueError):
            _PseudoLabelReader(reader, pseudo_labels, inductive=False)

    def test_negative_error(self):
        reader = rul_datasets.reader.DummyReader(1)
        pseudo_labels = [float(i) for i in range(-1, 9)]

        with pytest.raises(ValueError):
            _PseudoLabelReader(reader, pseudo_labels, inductive=False)

    def test_wrong_number_of_pseudo_labels_error(self):
        reader = rul_datasets.reader.DummyReader(1)
        pseudo_labels = [float(i) for i in range(30, 50)]  # expected number is 10
        pl_reader = _PseudoLabelReader(reader, pseudo_labels, inductive=False)

        with pytest.raises(RuntimeError):
            pl_reader.load_split("dev", "dev")

    @pytest.mark.parametrize(
        ["inductive", "exp_split"], [(True, "test"), (False, "dev")]
    )
    def test_pseudo_label_generation(self, inductive, exp_split):
        reader = rul_datasets.reader.DummyReader(1, percent_broken=0.8)
        pseudo_labels = [float(i) for i in range(40, 50)]
        pl_reader = _PseudoLabelReader(reader, pseudo_labels, inductive=inductive)

        features, targets = pl_reader.load_split(exp_split, "dev")
        for t, pl in zip(targets, pseudo_labels):
            assert t[-1] == pl
            assert np.all(t <= reader.max_rul)

    @pytest.mark.parametrize(
        ["inductive", "exp_split"], [(True, "test"), (False, "dev")]
    )
    @pytest.mark.parametrize("norm_rul", [True, False])
    def test_pseudo_label_generation_fttp(self, norm_rul, inductive, exp_split):
        fttp = 10
        reader = rul_datasets.reader.DummyReader(1, percent_broken=0.8)
        reader.first_time_to_predict = [fttp] * 30
        reader.norm_rul = norm_rul
        reader._preparator = mock.Mock(name="preperator")
        reader._preparator.run_split_dist = {
            "dev": list(range(1, 11)),
            "test": list(range(1, 11)),
        }
        pseudo_labels = [float(i) for i in range(40, 50)]
        pl_reader = _PseudoLabelReader(reader, pseudo_labels, inductive=inductive)

        features, targets = pl_reader.load_split(exp_split, "dev")
        for t, pl in zip(targets, pseudo_labels):
            if norm_rul:
                assert np.all(t <= 1.0)
            else:
                assert t[-1] == pl
            max_rul = t[0]
            assert np.all(t[:fttp] == max_rul)
            assert np.all(t[fttp:] < max_rul)
