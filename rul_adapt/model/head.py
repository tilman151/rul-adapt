from typing import List, Type, Optional

import torch
from torch import nn

from rul_adapt.utils import pairwise


class FullyConnectedHead(nn.Module):
    def __init__(
        self,
        input_channels: int,
        units: List[int],
        act_func: Type[nn.Module] = nn.ReLU,
        act_func_on_last_layer: bool = True,
    ) -> None:
        super().__init__()

        self.input_channels = input_channels
        self.units = units
        self.act_func = act_func
        self.act_func_on_last_layer = act_func_on_last_layer

        if not self.units:
            raise ValueError("Cannot build head network with no layers.")

        self._layers = self._get_layers()

    def _get_layers(self) -> nn.Module:
        units = [self.input_channels] + self.units
        act_funcs = [self.act_func] * len(self.units)
        act_funcs[-1] = self.act_func if self.act_func_on_last_layer else None

        layers = nn.Sequential()
        for (in_units, out_units), act_func in zip(pairwise(units), act_funcs):
            layers.append(self._get_fc_layer(in_units, out_units, act_func))

        return layers

    def _get_fc_layer(
        self, in_units: int, out_units: int, act_func: Optional[Type[nn.Module]]
    ) -> nn.Module:
        if act_func is None:
            layer = nn.Linear(in_units, out_units)
        else:
            layer = nn.Sequential(nn.Linear(in_units, out_units), act_func())

        return layer

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._layers(inputs)
