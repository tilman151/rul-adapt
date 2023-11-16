"""A module of feature extractors based on convolutional neural networks."""

from typing import List, Optional, Union, Type

import torch
from torch import nn

from rul_adapt import utils
from rul_adapt.utils import pairwise


class CnnExtractor(nn.Module):
    """A Convolutional Neural Network (CNN) based network that extracts a feature
    vector from same-length time windows.

    This feature extractor consists of multiple CNN layers and an optional fully
    connected (FC) layer. Each CNN layer can be configured with a number of filters
    and a kernel size. Additionally, batch normalization, same-padding and dropout
    can be applied. The fully connected layer can have a separate dropout
    probability.

    Both CNN and FC layers use ReLU activation functions by default. Custom
    activation functions can be set for each layer type.

    The data flow is as follows: `Input --> CNN x n --> [FC] --> Output`

    The expected input shape is `[batch_size, num_features, window_size]`. The output
    of this network is always flattened to `[batch_size, num_extracted_features]`.

    Examples:

        Without FC
        >>> import torch
        >>> from rul_adapt.model import CnnExtractor
        >>> cnn = CnnExtractor(14,units=[16, 1],seq_len=30)
        >>> cnn(torch.randn(10, 14, 30)).shape
        torch.Size([10, 26])

        With FC
        >>> import torch
        >>> from rul_adapt.model import CnnExtractor
        >>> cnn = CnnExtractor(14,units=[16, 1],seq_len=30,fc_units=16)
        >>> cnn(torch.randn(10, 14, 30)).shape
        torch.Size([10, 16])
    """

    def __init__(
        self,
        input_channels: int,
        units: List[int],
        seq_len: int,
        kernel_size: Union[int, List[int]] = 3,
        dilation: int = 1,
        stride: int = 1,
        padding: bool = False,
        fc_units: Optional[int] = None,
        dropout: float = 0.0,
        fc_dropout: float = 0.0,
        batch_norm: bool = False,
        act_func: Type[nn.Module] = nn.ReLU,
        fc_act_func: Type[nn.Module] = nn.ReLU,
    ):
        """
        Create a new CNN-based feature extractor.

        The `units` are the number of output filters for each CNN layer. The
        `seq_len` is needed to calculate the input units for the FC layer. The kernel
        size of each CNN layer can be set by passing a list to `kernel_size`. If an
        integer is passed, each layer has the same kernel size. If `padding` is true,
        same-padding is applied before each CNN layer, which keeps the window_size
        the same. If `batch_norm` is set, batch normalization is applied for each CNN
        layer. If `fc_units` is set, a fully connected layer is appended.

        Dropout can be applied to each CNN layer by setting `conv_dropout` to a
        number greater than zero. The same is valid for the fully connected layer and
        `fc_dropout`. Dropout will never be applied to the input layer.

        The whole network uses ReLU activation functions. This can be customized by
        setting either `conv_act_func` or `fc_act_func`.

        Args:
            input_channels: The number of input channels.
            units: The list of output filters for the CNN layers.
            seq_len: The window_size of the input data.
            kernel_size: The kernel size for the CNN layers. Passing an integer uses
                         the same kernel size for each layer.
            dilation: The dilation for the CNN layers.
            stride: The stride for the CNN layers.
            padding: Whether to apply same-padding before each CNN layer.
            fc_units: Number of output units for the fully connected layer.
            dropout: The dropout probability for the CNN layers.
            fc_dropout: The dropout probability for the fully connected layer.
            batch_norm: Whether to use batch normalization on the CNN layers.
            act_func: The activation function for the CNN layers.
            fc_act_func: The activation function for the fully connected layer.
        """
        super().__init__()

        self.input_channels = input_channels
        self.units = units
        self.seq_len = seq_len
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.padding = padding
        self.fc_units = fc_units
        self.dropout = dropout
        self.fc_dropout = fc_dropout
        self.batch_norm = batch_norm
        self.act_func = utils.str2callable(act_func, restriction="torch.nn")
        self.fc_act_func = utils.str2callable(fc_act_func, restriction="torch.nn")

        self._kernel_sizes = (
            [self.kernel_size] * len(self.units)
            if isinstance(self.kernel_size, int)
            else self.kernel_size
        )
        self._layers = self._get_layers()

    def _get_layers(self) -> nn.Module:
        layers = nn.Sequential()
        filter_iter = pairwise([self.input_channels] + self.units)
        layer_iter = zip(filter_iter, self._kernel_sizes)
        for i, ((in_ch, out_ch), ks) in enumerate(layer_iter):
            layers.add_module(f"conv_{i}", self._get_conv_layer(in_ch, out_ch, ks, i))

        layers.append(nn.Flatten())
        if self.fc_units is not None:
            flat_dim = self._get_flat_dim()
            layers.add_module("fc", self._get_fc_layer(flat_dim, self.fc_units))

        return layers

    def _get_conv_layer(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        num_layer: int,
    ) -> nn.Module:
        conv_layer = nn.Sequential()
        if num_layer > 0 and self.dropout > 0:
            conv_layer.append(nn.Dropout1d(self.dropout))
        conv_layer.append(
            nn.Conv1d(
                input_channels,
                output_channels,
                kernel_size,
                dilation=self.dilation,
                stride=self.stride,
                bias=not self.batch_norm,
                padding=self._get_padding(),
            )
        )
        if self.batch_norm:
            conv_layer.append(nn.BatchNorm1d(output_channels))
        conv_layer.append(self.act_func())

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
        after_conv = self.seq_len
        for kernel_size in self._kernel_sizes:
            cut_off = 0 if self.padding else self.dilation * (kernel_size - 1)
            after_conv = (after_conv - cut_off - 1) // self.stride + 1
        flat_dim = after_conv * self.units[-1]

        return flat_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._layers(inputs)
