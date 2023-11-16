"""A module of feature extractors based on recurrent neural networks."""

from typing import List, Optional, Tuple, Type, Union

import torch
from torch import nn

from rul_adapt.utils import pairwise


class LstmExtractor(nn.Module):
    """A Long Short Term Memory (LSTM) based network that extracts a feature vector
    from same-length time windows.

    This feature extractor consists of a multi-layer LSTM and an optional fully
    connected (FC) layer with a ReLU activation function. The LSTM layers can be
    configured as bidirectional. Dropout can be applied separately to LSTM and FC
    layers.

    The data flow is as follows: `Input --> LSTM x n --> [FC] --> Output`

    The expected input shape is `[batch_size, num_features, window_size]`.

    Examples:

        Without FC
        >>> import torch
        >>> from rul_adapt.model import LstmExtractor
        >>> lstm = LstmExtractor(input_channels=14,units=[16, 16])
        >>> lstm(torch.randn(10, 14, 30)).shape
        torch.Size([10, 16])

        With FC
        >>> from rul_adapt.model import LstmExtractor
        >>> lstm = LstmExtractor(input_channels=14,units=[16, 16],fc_units=8)
        >>> lstm(torch.randn(10, 14, 30)).shape
        torch.Size([10, 8])
    """

    def __init__(
        self,
        input_channels: int,
        units: List[int],
        fc_units: Optional[int] = None,
        dropout: float = 0.0,
        fc_dropout: float = 0.0,
        bidirectional: bool = False,
    ) -> None:
        """
        Create a new LSTM-based feature extractor.

        The `units` are the output units for each LSTM layer. If `bidirectional`
        is set to `True`, a BiLSTM is used and the output units are doubled. If
        `fc_units` is set, a fully connected layer is appended. The number of output
        features of this network is either `units[-1]` by default,
        `2 * units[ -1]` if bidirectional is set, or `fc_units` if it is set.

        Dropout can be applied to each LSTM layer by setting `lstm_dropout` to a
        number greater than zero. The same is valid for the fully connected layer and
        `fc_dropout`.

        Args:
            input_channels: The number of input channels.
            units: The list of output units for the LSTM layers.
            fc_units: The number of output units for the fully connected layer.
            dropout: The dropout probability for the LSTM layers.
            fc_dropout: The dropout probability for the fully connected layer.
            bidirectional: Whether to use a BiLSTM.
        """
        super().__init__()

        self.input_channels = input_channels
        self.units = units
        self.dropout = dropout
        self.fc_units = fc_units
        self.fc_dropout = fc_dropout
        self.bidirectional = bidirectional

        self._lstm_layers = _get_rnn_layers(
            nn.LSTM,
            self.units,
            self.input_channels,
            self.dropout,
            self.bidirectional,
        )
        self._fc_layer = self._get_fc_layer()

    def _get_fc_layer(self) -> nn.Module:
        fc_layer = nn.Sequential()
        if self.fc_units is not None:
            if self.fc_dropout > 0:
                fc_layer.append(nn.Dropout(self.fc_dropout))
            fc_in_units = self.units[-1] * (2 if self.bidirectional else 1)
            fc_layer.append(nn.Linear(fc_in_units, self.fc_units))
            fc_layer.append(nn.ReLU())

        return fc_layer

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = torch.permute(inputs, (2, 0, 1))
        inputs, _ = self._lstm_layers(inputs)
        if self.bidirectional:
            forward, backward = torch.split(inputs, self.units[-1], dim=2)
            inputs = torch.cat([forward[-1], backward[0]], dim=1)
        else:
            inputs = inputs[-1]
        inputs = self._fc_layer(inputs)

        return inputs


class GruExtractor(nn.Module):
    """A Gated Recurrent Unit (GRU) based network that extracts a feature vector
    from same-length time windows.

    This feature extractor consists of multiple fully connected (FC) layers with
    a ReLU activation functions and a multi-layer GRU. The GRU layers can be
    configured as bidirectional. Dropout can be applied separately to the GRU
    layers.

    The data flow is as follows: `Input --> FC x n --> GRU x m --> Output`

    The expected input shape is `[batch_size, num_features, window_size]`.

    Examples:
        >>> import torch
        >>> from rul_adapt.model import GruExtractor
        >>> gru = GruExtractor(input_channels=14, fc_units=[16, 8], gru_units=[8])
        >>> gru(torch.randn(10, 14, 30)).shape
        torch.Size([10, 8])
    """

    def __init__(
        self,
        input_channels: int,
        fc_units: List[int],
        gru_units: List[int],
        gru_dropout: float = 0.0,
        bidirectional: bool = False,
    ) -> None:
        """
        Create a new GRU-based feature extractor.

        The `fc_units` are the output units for each fully connected layer
        and `gru_units` for each LSTM layer. If `bidirectional` is set to `True`,
        a BiGRU is used and the output units are doubled. The number of output
        features of this network is either `gru_units[-1]` by default, or `2 *
        gru_units[ -1]` if bidirectional is set.

        Dropout can be applied to each GRU layer by setting `lstm_dropout` to a
        number greater than zero.

        Args:
            input_channels: The number of input channels.
            fc_units: The list of output units for the fully connected layers.
            gru_units: The list of output units for the GRU layers
            gru_dropout: The dropout probability for the GRU layers.
            bidirectional: Whether to use a BiGRU.
        """
        super().__init__()

        self.input_channels = input_channels
        self.gru_units = gru_units
        self.gru_dropout = gru_dropout
        self.fc_units = fc_units
        self.bidirectional = bidirectional

        self._fc_layer = self._get_fc_layers()
        self._gru_layers = _get_rnn_layers(
            nn.GRU,
            self.gru_units,
            self.fc_units[-1],
            self.gru_dropout,
            self.bidirectional,
        )

    def _get_fc_layers(self) -> nn.Module:
        fc_layer = nn.Sequential()
        fc_units = [self.input_channels] + self.fc_units
        for in_units, out_units in pairwise(fc_units):
            fc_layer.append(nn.Conv1d(in_units, out_units, kernel_size=1))
            fc_layer.append(nn.ReLU())

        return fc_layer

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self._fc_layer(inputs)  # 1x1 conv as linear layer on each time step
        inputs = torch.permute(inputs, (2, 0, 1))
        inputs, _ = self._gru_layers(inputs)
        if self.bidirectional:
            forward, backward = torch.split(inputs, self.gru_units[-1], dim=2)
            inputs = torch.cat([forward[-1], backward[0]], dim=1)
        else:
            inputs = inputs[-1]

        return inputs


def _get_rnn_layers(
    rnn_func: Union[Type[nn.LSTM], Type[nn.GRU]],
    num_units: List[int],
    input_channels: int,
    dropout: float,
    bidirectional: bool,
) -> nn.Module:
    layers: nn.Module
    if all(num_units[0] == u for u in num_units):
        layers = rnn_func(
            input_channels,
            num_units[0],
            num_layers=len(num_units),
            dropout=dropout,
            bidirectional=bidirectional,
        )
    else:
        layers = _Rnn(rnn_func, input_channels, num_units, dropout, bidirectional)

    return layers


class _Rnn(nn.Module):
    def __init__(
        self,
        rnn_func: Union[Type[nn.LSTM], Type[nn.GRU]],
        input_channels: int,
        rnn_units: List[int],
        dropout: float,
        bidirectional: bool,
    ) -> None:
        super().__init__()

        self._rnn_func = rnn_func
        self.input_channels = input_channels
        self.rnn_units = rnn_units
        self.dropout = dropout
        self.bidirectional = bidirectional

        if not self.rnn_units:
            raise ValueError(f"Cannot build stacked {self._rnn_func} with zero layers.")
        self._layers = self._get_layers()

    def _get_layers(self) -> nn.ModuleList:
        lstm_layers = nn.ModuleList()  # registers items as parameters
        #
        unit_multiplier = 2 if self.bidirectional else 1  # 2x in channels for BiRnn
        unit_iter = pairwise([self.input_channels / unit_multiplier] + self.rnn_units)
        for input_channels, num_units in unit_iter:
            lstm_layers.append(
                self._rnn_func(
                    int(input_channels * unit_multiplier),  # in channels could be odd
                    num_units,
                    bidirectional=self.bidirectional,
                )
            )

        return lstm_layers

    def forward(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        for i, rnn in enumerate(self._layers, start=1):
            inputs, states = rnn(inputs)
            if i < len(self._layers):  # no dropout on last layer outputs
                inputs = nn.functional.dropout(inputs, self.dropout)

        return inputs, states
