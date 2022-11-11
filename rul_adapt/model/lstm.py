from itertools import pairwise
from typing import List, Optional, Tuple

import torch
from torch import nn


class LstmExtractor(nn.Module):
    def __init__(
        self,
        input_channels: int,
        lstm_units: List[int],
        fc_units: Optional[int] = None,
        lstm_dropout: float = 0.0,
        fc_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.input_channels = input_channels
        self.lstm_units = lstm_units
        self.lstm_dropout = lstm_dropout
        self.fc_units = fc_units
        self.fc_dropout = fc_dropout

        self._lstm_layers = self._get_lstm_layers()
        self._fc_layer = self._get_fc_layer()

    def _get_lstm_layers(self) -> nn.Module:
        lstm_layers: nn.Module
        if all(self.lstm_units[0] == u for u in self.lstm_units):
            lstm_layers = nn.LSTM(
                self.input_channels,
                self.lstm_units[0],
                num_layers=len(self.lstm_units),
                dropout=self.lstm_dropout,
            )
        else:
            lstm_layers = _Lstm(self.input_channels, self.lstm_units, self.lstm_dropout)

        return lstm_layers

    def _get_fc_layer(self) -> nn.Module:
        fc_layer = nn.Sequential()
        if self.fc_units is not None:
            if self.fc_dropout > 0:
                fc_layer.append(nn.Dropout(self.fc_dropout))
            fc_layer.append(nn.Linear(self.lstm_units[-1], self.fc_units))
            fc_layer.append(nn.ReLU())

        return fc_layer

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = torch.permute(inputs, (2, 0, 1))
        inputs, _ = self._lstm_layers(inputs)
        inputs = inputs[-1]  # get last time step
        inputs = self._fc_layer(inputs)

        return inputs


class _Lstm(nn.Module):
    def __init__(
        self, input_channels: int, lstm_units: List[int], dropout: float
    ) -> None:
        super().__init__()

        self.input_channels = input_channels
        self.lstm_units = lstm_units
        self.dropout = dropout

        if not self.lstm_units:
            raise ValueError("Cannot build stacked LSTM with zero layers.")
        self._lstm_layers = self._get_lstm_layers()

    def _get_lstm_layers(self) -> nn.ModuleList:
        lstm_layers = nn.ModuleList()  # registers items as parameters
        unit_iter = pairwise([self.input_channels] + self.lstm_units)
        for input_channels, num_units in unit_iter:
            lstm_layers.append(
                nn.LSTM(
                    input_channels,
                    num_units,
                    dropout=self.dropout,
                )
            )

        return lstm_layers

    def forward(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        for lstm in self._lstm_layers:
            inputs, states = lstm(inputs)

        return inputs, states
