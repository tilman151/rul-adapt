import pytest

import rul_adapt.construct


@pytest.mark.parametrize("source_fd", [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize("target_fd", [0, 1, 2, 3, 4, 5])
def test_get_lstm_dann(source_fd, target_fd):
    if source_fd == target_fd:
        with pytest.raises(ValueError):
            rul_adapt.construct.get_lstm_dann(source_fd, target_fd)
    elif 1 > source_fd or source_fd > 4 or 1 > target_fd or target_fd > 4:
        with pytest.raises(ValueError):
            rul_adapt.construct.get_lstm_dann(source_fd, target_fd)
    else:
        rul_adapt.construct.get_lstm_dann(source_fd, target_fd)
