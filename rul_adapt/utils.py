from itertools import tee
from typing import Union, Callable

import torchmetrics


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def str2callable(cls: Union[str, Callable], restriction: str = "") -> Callable:
    """Dynamically import a callable from a string."""
    if isinstance(cls, Callable):  # type: ignore[arg-type]
        return cls  # type: ignore[return-value]
    if isinstance(cls, str) and not cls.startswith(restriction):
        raise ValueError(
            f"Failed to import '{cls}' because "
            f"imports are restricted to '{restriction}'."
        )

    module_name, class_name = cls.rsplit(".", 1)  # type: ignore[union-attr]
    module = __import__(module_name, fromlist=[class_name])
    cls = getattr(module, class_name)

    return cls  # type: ignore[return-value]


def get_loss(loss_type: str) -> torchmetrics.Metric:
    if loss_type == "mae":
        train_source_loss = torchmetrics.MeanAbsoluteError()
    elif loss_type == "mse":
        train_source_loss = torchmetrics.MeanSquaredError()
    elif loss_type == "rmse":
        train_source_loss = torchmetrics.MeanSquaredError(squared=False)
    else:
        raise ValueError(
            f"Unknown loss type '{loss_type}'. " "Use either 'mae', 'mse' or 'rmse'."
        )

    return train_source_loss
