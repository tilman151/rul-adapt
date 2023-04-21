from typing import Type

import torch
from torch import nn


class ActivationDropoutWrapper(nn.Module):
    def __init__(
        self,
        wrapped: nn.Module,
        act_func: Type[nn.Module] = nn.ReLU,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.wrapped = wrapped
        self.act_func = act_func
        self.dropout = dropout

        self._act_func = self.act_func()
        self._dropout = nn.Dropout(self.dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._dropout(self._act_func(self.wrapped(inputs)))


class DropoutPrefix(nn.Module):
    def __init__(
        self,
        wrapped: nn.Module,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.wrapped = wrapped
        self.dropout = dropout

        self._dropout = nn.Dropout(self.dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.wrapped(self._dropout(inputs))
