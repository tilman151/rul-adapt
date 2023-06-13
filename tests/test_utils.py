import pytest
import torch
from torch import nn

from rul_adapt.utils import str2callable, OptimizerFactory


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


@pytest.fixture()
def parameters():
    return [nn.Parameter(torch.randn(10, 10)) for _ in range(3)]


class TestOptimizerFactory:
    @pytest.mark.parametrize(
        ["optim_type", "optim"],
        [
            ("adam", torch.optim.Adam),
            ("sgd", torch.optim.SGD),
            ("rmsprop", torch.optim.RMSprop),
            pytest.param("foo", None, marks=pytest.mark.xfail(strict=True)),
        ],
    )
    def test_optim_type(self, parameters, optim_type, optim):
        factory = OptimizerFactory(optim_type=optim_type)
        assert isinstance(factory(parameters)["optimizer"], optim)

    def test_lr(self, parameters):
        factory = OptimizerFactory(optim_type="adam", lr=0.1)
        assert factory(parameters)["optimizer"].defaults["lr"] == 0.1

    def test_optim_kwargs(self, parameters):
        factory = OptimizerFactory(optim_type="adam", optim_weight_decay=0.1)
        assert factory(parameters)["optimizer"].defaults["weight_decay"] == 0.1

    @pytest.mark.parametrize(
        ["scheduler_type", "scheduler", "kwargs"],
        [
            ("step", torch.optim.lr_scheduler.StepLR, {"step_size": 100}),
            ("cosine", torch.optim.lr_scheduler.CosineAnnealingLR, {"T_max": 100}),
            ("linear", torch.optim.lr_scheduler.LinearLR, {}),
            ("lambda", torch.optim.lr_scheduler.LambdaLR, {"lr_lambda": lambda x: x}),
            (None, None, {}),
        ],
    )
    def test_scheduler_type(self, parameters, scheduler_type, scheduler, kwargs):
        kwargs = {f"scheduler_{k}": v for k, v in kwargs.items()}
        factory = OptimizerFactory(
            optim_type="adam", scheduler_type=scheduler_type, **kwargs
        )
        if scheduler_type is None:
            assert "lr_scheduler" not in factory(parameters)
        else:
            assert isinstance(
                factory(parameters)["lr_scheduler"]["scheduler"], scheduler
            )

    def test_scheduler_kwargs(self, parameters):
        factory = OptimizerFactory(
            optim_type="adam", scheduler_type="step", scheduler_step_size=100
        )
        assert factory(parameters)["lr_scheduler"]["scheduler"].step_size == 100

    def test_excess_kwargs_warning(self):
        with pytest.warns(UserWarning, match=r"will be ignored: \['foo'\]"):
            OptimizerFactory(optim_type="adam", foo=1)
