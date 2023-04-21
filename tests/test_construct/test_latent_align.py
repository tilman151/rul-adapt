import pytest

import rul_adapt


@pytest.mark.parametrize("source_fd", [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize("target_fd", [0, 1, 2, 3, 4, 5])
def test_get_latent_align_cmapss(source_fd, target_fd):
    _check_config("cmapss", source_fd, target_fd, lower=1, upper=4)


@pytest.mark.parametrize("target_fd", [0, 1, 2, 3])
@pytest.mark.parametrize("source_fd", [0, 1, 2, 3])
@pytest.mark.parametrize("subtask", [1, 2])
def test_get_latent_align_xjtu_sy(source_fd, target_fd, subtask):
    _check_config("xjtu-sy", source_fd, target_fd, subtask, lower=1, upper=2)


def _check_config(dataset, source_fd, target_fd, subtask=None, lower=1, upper=4):
    if source_fd == target_fd:
        with pytest.raises(ValueError):
            rul_adapt.construct.get_latent_align(dataset, source_fd, target_fd)
    elif (
        lower > source_fd or source_fd > upper or lower > target_fd or target_fd > upper
    ):
        with pytest.raises(ValueError):
            rul_adapt.construct.get_latent_align(dataset, source_fd, target_fd, subtask)
    else:
        rul_adapt.construct.get_latent_align(dataset, source_fd, target_fd, subtask)


def test_get_latent_align_trainer_override():
    _, _, trainer = rul_adapt.construct.get_latent_align(
        "cmapss", 1, 2, max_epochs=6, min_epochs=5
    )
    assert trainer.max_epochs == 6  # override existing option
    assert trainer.min_epochs == 5  # override additional option
