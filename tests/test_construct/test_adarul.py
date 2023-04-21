import pytest

import rul_adapt.construct


@pytest.mark.parametrize("source_fd", [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize("target_fd", [0, 1, 2, 3, 4, 5])
def test_get_adarul(source_fd, target_fd):
    if source_fd == target_fd:
        with pytest.raises(ValueError):
            rul_adapt.construct.get_adarul(source_fd, target_fd)
    elif 1 > source_fd or source_fd > 4 or 1 > target_fd or target_fd > 4:
        with pytest.raises(ValueError):
            rul_adapt.construct.get_adarul(source_fd, target_fd)
    else:
        rul_adapt.construct.get_adarul(source_fd, target_fd)


def test_get_adarul_trainer_override():
    (_, _, pre_trainer), (_, _, _, trainer) = rul_adapt.construct.get_adarul(
        1,
        2,
        {"max_epochs": 5, "min_epochs": 4},
        {"max_epochs": 6, "min_epochs": 5},
    )
    assert pre_trainer.max_epochs == 5  # override existing option
    assert pre_trainer.min_epochs == 4  # override additional option
    assert trainer.max_epochs == 6  # override existing option
    assert trainer.min_epochs == 5  # override additional option
