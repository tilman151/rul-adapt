import pytest

from rul_adapt.utils import str2callable


@pytest.mark.parametrize(
    ["inputs", "expected", "restriction"],
    [
        ("rul_adapt.utils.str2callable", str2callable, ""),
        (str2callable, str2callable, ""),
        ("rul_adapt.utils.str2callable", str2callable, "rul_adapt.utils"),
        (str2callable, str2callable, "rul_adapt.utils"),
        pytest.param(
            "rul_adapt.utils.str2callable",
            str2callable,
            "torch",
            marks=pytest.mark.xfail(strict=True),
        ),
    ],
)
def test_str2callable(inputs, expected, restriction):
    assert str2callable(inputs, restriction) is expected
