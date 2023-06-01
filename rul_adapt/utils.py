from itertools import tee
from typing import Union, Callable, Literal

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
    """Get a loss instance by specifying a string."""
    loss: torchmetrics.Metric
    if loss_type == "mae":
        loss = torchmetrics.MeanAbsoluteError()
    elif loss_type == "mse":
        loss = torchmetrics.MeanSquaredError()
    elif loss_type == "rmse":
        loss = torchmetrics.MeanSquaredError(squared=False)
    else:
        raise ValueError(
            f"Unknown loss type '{loss_type}'. " "Use either 'mae', 'mse' or 'rmse'."
        )

    return loss


def dataloader2domain(dataloader_idx: int) -> Literal["source", "target"]:
    if dataloader_idx == 0:
        return "source"
    elif dataloader_idx == 1:
        return "target"
    else:
        raise RuntimeError(
            f"Expected dataloader_idx to be 0 or 1 but got {dataloader_idx}."
        )
