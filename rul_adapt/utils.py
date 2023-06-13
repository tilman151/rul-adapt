import warnings
from itertools import tee
from typing import Union, Callable, Literal, Any, Dict, Optional, Iterable

import torch
import torchmetrics
from torch import nn


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


class OptimizerFactory:
    """Factory for creating optimizers and schedulers.

    After initialization, the factory can be called to create an optimizer with an
    optional scheduler."""

    def __init__(
        self,
        optim_type: str = "adam",
        lr: float = 1e-3,
        scheduler_type: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Create a new factory to efficiently create optimizers and schedulers.

        The factory creates an optimizer of the specified `optim_type` and adds an
        optional scheduler of the specified `scheduler_type`. Additional keyword
        arguments for the optimizer can be passed by adding the 'optim_' prefix and
        for the scheduler by adding the 'scheduler_' prefix. The factory will ignore
        any other keyword arguments.

        Available optimizers are 'adam', 'sgd' and 'rmsprop'. Available schedulers
        are 'step', 'cosine', 'linear' and 'lambda'.

        Args:
            optim_type: The type of optimizer to create.
            lr: The learning rate to use.
            scheduler_type: The optional type of scheduler to create.
            **kwargs: Additional keyword arguments for the optimizer and scheduler.
        """
        self.optim_type = optim_type
        self.lr = lr
        self.scheduler_type = scheduler_type
        self._kwargs = kwargs

        self._warn_excess_kwargs()

    def _warn_excess_kwargs(self) -> None:
        def _is_excess_kwarg(key: str) -> bool:
            return not (key.startswith("optim_") or key.startswith("scheduler_"))

        excess_kwargs = list(filter(_is_excess_kwarg, self._kwargs.keys()))
        if excess_kwargs:
            warnings.warn(
                "The following kwargs were passed but do not start "
                "with 'optim_' or 'scheduler_' and therefore "
                f"will be ignored: {excess_kwargs}."
            )

    def __call__(self, parameters: Iterable[nn.Parameter]) -> Dict[str, Any]:
        """
        Create an optimizer with an optional scheduler for the given parameters.

        The object returned by this method is a lightning optimizer config and can be
        the return value of `configure_optimizers`.

        Args:
            parameters: The model parameters to optimize.

        Returns:
            A lightning optimizer config.
        """
        optim_kwargs = {
            key.replace("optim_", ""): value
            for key, value in self._kwargs.items()
            if key.startswith("optim_")
        }
        optim = self._optim_func(parameters, lr=self.lr, **optim_kwargs)
        optim_config = {"optimizer": optim}

        if self.scheduler_type is not None:
            scheduler_kwargs = {
                key.replace("scheduler_", ""): value
                for key, value in self._kwargs.items()
                if key.startswith("scheduler_")
            }
            optim_config["lr_scheduler"] = {
                "scheduler": self._scheduler_func(optim, **scheduler_kwargs)
            }

        return optim_config

    @property
    def _optim_func(self) -> Callable:
        if self.optim_type == "adam":
            return torch.optim.Adam
        elif self.optim_type == "sgd":
            return torch.optim.SGD
        elif self.optim_type == "rmsprop":
            return torch.optim.RMSprop
        else:
            raise ValueError(
                f"Unknown optimizer type '{self.optim_type}'. "
                "Use either 'adam', 'sgd' or 'rmsprop'."
            )

    @property
    def _scheduler_func(self) -> Callable:
        if self.scheduler_type == "step":
            return torch.optim.lr_scheduler.StepLR
        elif self.scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR
        elif self.scheduler_type == "linear":
            return torch.optim.lr_scheduler.LinearLR
        elif self.scheduler_type == "lambda":
            return torch.optim.lr_scheduler.LambdaLR
        else:
            raise ValueError(
                f"Unknown scheduler type '{self.scheduler_type}'. "
                "Use either 'step', 'cosine', 'linear' or 'lambda'."
            )
