from typing import List, Optional, Union, Type

import torch
from torch import nn

from rul_adapt.utils import pairwise


class CnnExtractor(nn.Module):
    def __init__(
        self,
        input_channels: int,
        conv_filters: List[int],
        seq_len: int,
        kernel_size: Union[int, List[int]] = 3,
        padding: bool = False,
        fc_units: Optional[int] = None,
        conv_dropout: float = 0.0,
        fc_dropout: float = 0.0,
        batch_norm: bool = False,
        conv_act_func: Type[nn.Module] = nn.ReLU,
        fc_act_func: Type[nn.Module] = nn.ReLU,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.conv_filters = conv_filters
        self.seq_len = seq_len
        self.kernel_size = kernel_size
        self.padding = padding
        self.fc_units = fc_units
        self.conv_dropout = conv_dropout
        self.fc_dropout = fc_dropout
        self.batch_norm = batch_norm
        self.conv_act_func = conv_act_func
        self.fc_act_func = fc_act_func

        self._kernel_sizes = (
            [self.kernel_size] * len(self.conv_filters)
            if isinstance(self.kernel_size, int)
            else self.kernel_size
        )
        self._layers = self._get_layers()

    def _get_layers(self) -> nn.Module:
        layers = nn.Sequential()
        filter_iter = pairwise([self.input_channels] + self.conv_filters)
        for i, (input_channels, output_channels) in enumerate(filter_iter):
            kernel_size = self._kernel_sizes[i]
            layers.add_module(
                f"conv_{i}",
                self._get_conv_layer(input_channels, output_channels, kernel_size),
            )
        layers.append(nn.Flatten())
        if self.fc_units is not None:
            layers.add_module(
                "fc", self._get_fc_layer(self._get_flat_dim(), self.fc_units)
            )

        return layers

    def _get_conv_layer(
        self, input_channels: int, output_channels: int, kernel_size: int
    ) -> nn.Module:
        conv_layer = nn.Sequential()
        if self.conv_dropout > 0:
            conv_layer.append(nn.Dropout1d(self.conv_dropout))
        conv_layer.append(
            nn.Conv1d(
                input_channels,
                output_channels,
                kernel_size,
                bias=not self.batch_norm,
                padding=self._get_padding(),
            )
        )
        if self.batch_norm:
            conv_layer.append(nn.BatchNorm1d(output_channels))
        conv_layer.append(self.conv_act_func())

        return conv_layer

    def _get_fc_layer(self, input_units: int, output_units: int) -> nn.Module:
        fc_layer = nn.Sequential()
        if self.fc_dropout > 0:
            fc_layer.append(nn.Dropout(self.fc_dropout))
        fc_layer.append(nn.Linear(input_units, output_units))
        fc_layer.append(self.fc_act_func())

        return fc_layer

    def _get_padding(self) -> str:
        return "same" if self.padding else "valid"

    def _get_flat_dim(self) -> int:
        if self.padding:
            less_per_conv = [0] * len(self.conv_filters)
        else:
            less_per_conv = [ks - 1 for ks in self._kernel_sizes]
        after_conv = self.seq_len - sum(less_per_conv)
        flat_dim = after_conv * self.conv_filters[-1]

        return flat_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._layers(inputs)
