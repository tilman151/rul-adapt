from typing import List, Optional, Tuple, Type, Union

import torch
from torch import nn

from rul_adapt.utils import pairwise


class LstmExtractor(nn.Module):
    def __init__(
        self,
        input_channels: int,
        lstm_units: List[int],
        fc_units: Optional[int] = None,
        lstm_dropout: float = 0.0,
        fc_dropout: float = 0.0,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()

        self.input_channels = input_channels
        self.lstm_units = lstm_units
        self.lstm_dropout = lstm_dropout
        self.fc_units = fc_units
        self.fc_dropout = fc_dropout
        self.bidirectional = bidirectional

        self._lstm_layers = _get_rnn_layers(
            nn.LSTM,
            self.lstm_units,
            self.input_channels,
            self.lstm_dropout,
            self.bidirectional,
        )
        self._fc_layer = self._get_fc_layer()

    def _get_fc_layer(self) -> nn.Module:
        fc_layer = nn.Sequential()
        if self.fc_units is not None:
            if self.fc_dropout > 0:
                fc_layer.append(nn.Dropout(self.fc_dropout))
            fc_in_units = self.lstm_units[-1] * (2 if self.bidirectional else 1)
            fc_layer.append(nn.Linear(fc_in_units, self.fc_units))
            fc_layer.append(nn.ReLU())

        return fc_layer

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = torch.permute(inputs, (2, 0, 1))
        inputs, _ = self._lstm_layers(inputs)
        if self.bidirectional:
            forward, backward = torch.split(inputs, self.lstm_units[-1], dim=2)
            inputs = torch.cat([forward[-1], backward[0]], dim=1)
        else:
            inputs = inputs[-1]
        inputs = self._fc_layer(inputs)

        return inputs


class GruExtractor(nn.Module):
    def __init__(
        self,
        input_channels: int,
        fc_units: List[int],
        gru_units: List[int],
        gru_dropout: float = 0.0,
        bidirectional: bool = False,
    ) -> None:
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
                    dropout=self.dropout,
                    bidirectional=self.bidirectional,
                )
            )

        return lstm_layers

    def forward(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        for rnn in self._layers:
            inputs, states = rnn(inputs)

        return inputs, states
