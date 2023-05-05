from unittest import mock

import pytest
import rul_datasets
import torch

from rul_adapt import model
from rul_adapt.approach import SupervisedApproach
from rul_adapt.approach.pseudo_labels import generate_pseudo_labels


def test_generate_pseudo_labels():
    fe = model.CnnExtractor(1, [3], 10, fc_units=16)
    reg = model.FullyConnectedHead(16, [1], act_func_on_last_layer=False)
    approach = SupervisedApproach(0.1, "mse", "adam")
    approach.set_model(fe, reg)
    dm = rul_datasets.RulDataModule(rul_datasets.reader.DummyReader(1), 32)

    pseudo_labels = generate_pseudo_labels(dm, approach)

    assert len(pseudo_labels) == 10  # number of runs in dev split of dummy dataset
    assert all([isinstance(pl, float) for pl in pseudo_labels])


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
