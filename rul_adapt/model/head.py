"""A module for network working as a regression or classification head."""

from typing import List, Type, Optional

import torch
from torch import nn

from rul_adapt import utils
from rul_adapt.utils import pairwise


class FullyConnectedHead(nn.Module):
    """A fully connected (FC) network that can be used as a RUL regressor or a domain
    discriminator.

    This network is a stack of fully connected layers with ReLU activation functions
    by default. The activation function can be customized through the `act_func`
    parameter. If the last layer of the network should not have an activation
    function, `act_func_on_last_layer` can be set to `False`.

    The data flow is as follows: `Inputs --> FC x n --> Outputs`

    The expected input shape is `[batch_size, num_features]`.

    Examples:
        Default
        >>> import torch
        >>> from rul_adapt.model import FullyConnectedHead
        >>> regressor = FullyConnectedHead(32, [16, 1])
        >>> outputs = regressor(torch.randn(10, 32))
        >>> outputs.shape
        torch.Size([10, 1])
        >>> type(outputs.grad_fn)
        <class 'ReluBackward0'>

        Custom activation function
        >>> import torch
        >>> from rul_adapt.model import FullyConnectedHead
        >>> regressor = FullyConnectedHead(32, [16, 1], act_func=torch.nn.Sigmoid)
        >>> outputs = regressor(torch.randn(10, 32))
        >>> type(outputs.grad_fn)
        <class 'SigmoidBackward0'>

        Without activation function on last layer
        >>> import torch
        >>> from rul_adapt.model import FullyConnectedHead
        >>> regressor = FullyConnectedHead(32, [16, 1], act_func_on_last_layer=False)
        >>> outputs = regressor(torch.randn(10, 32))
        >>> type(outputs.grad_fn)
        <class 'AddmmBackward0'>
    """

    def __init__(
        self,
        input_channels: int,
        units: List[int],
        dropout: float = 0.0,
        act_func: Type[nn.Module] = nn.ReLU,
        act_func_on_last_layer: bool = True,
    ) -> None:
        """
        Create a new fully connected head network.

        The `units` are the number of output units for each FC layer. The number of
        output features is `units[-1]`. If dropout is used, it is applied in *each*
        layer, including input.

        Args:
            input_channels: The number of input channels.
            units: The number of output units for the FC layers.
            dropout: The dropout probability before each layer. Set to zero to
                     deactivate.
            act_func: The activation function for each layer.
            act_func_on_last_layer: Whether to add the activation function to the last
                                    layer.
        """
        super().__init__()

        self.input_channels = input_channels
        self.units = units
        self.dropout = dropout
        self.act_func: Type[nn.Module] = utils.str2callable(  # type: ignore[assignment]
            act_func, restriction="torch.nn"
        )
        self.act_func_on_last_layer = act_func_on_last_layer

        if not self.units:
            raise ValueError("Cannot build head network with no layers.")

        self._layers = self._get_layers()

    def _get_layers(self) -> nn.Module:
        units = [self.input_channels] + self.units
        act_funcs: List[Optional[Type[nn.Module]]] = [self.act_func] * len(self.units)
        act_funcs[-1] = self.act_func if self.act_func_on_last_layer else None

        layers = nn.Sequential()
        for (in_units, out_units), act_func in zip(pairwise(units), act_funcs):
            layers.append(self._get_fc_layer(in_units, out_units, act_func))

        return layers

    def _get_fc_layer(
        self, in_units: int, out_units: int, act_func: Optional[Type[nn.Module]]
    ) -> nn.Module:
        layer: nn.Module
        if self.dropout > 0:
            layer = nn.Sequential(nn.Dropout(self.dropout))
        else:
            layer = nn.Sequential()

        layer.append(nn.Linear(in_units, out_units))
        if act_func is not None:
            layer.append(act_func())

        return layer

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._layers(inputs)
