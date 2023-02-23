from unittest import mock

import pytest

import rul_adapt.construct


@pytest.mark.parametrize("source_fd", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("target_fd", [0, 1, 2, 3, 4])
@mock.patch(
    "rul_datasets.reader.FemtoReader.load_complete_split", return_value=([], [])
)
@mock.patch("rul_datasets.reader.FemtoReader.prepare_data")
def test_get_lstm_dann(_, __, source_fd, target_fd):
    """Reader is mocked to avoid loading data because VibrationFeatureExtractor is
    fitted on it."""
    if source_fd == target_fd:
        with pytest.raises(ValueError):
            rul_adapt.construct.get_tbigru(source_fd, target_fd)
    elif 1 > source_fd or source_fd > 3 or 1 > target_fd or target_fd > 3:
        with pytest.raises(ValueError):
            rul_adapt.construct.get_tbigru(source_fd, target_fd)
    else:
        rul_adapt.construct.get_tbigru(source_fd, target_fd)


@mock.patch(
    "rul_datasets.reader.FemtoReader.load_complete_split", return_value=([], [])
)
@mock.patch("rul_datasets.reader.FemtoReader.prepare_data")
def test_get_lstm_dann_trainer_override(_, __):
    """Reader is mocked to avoid loading data because VibrationFeatureExtractor is
    fitted on it."""
    _, _, trainer = rul_adapt.construct.get_tbigru(1, 2, max_epochs=5, min_epochs=4)
    assert trainer.max_epochs == 5  # override existing option
    assert trainer.min_epochs == 4  # override additional option
