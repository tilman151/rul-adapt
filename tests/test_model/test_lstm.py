import pytest
import torch
from torch import nn

from rul_adapt.model.lstm import LstmExtractor


@pytest.fixture
def inputs():
    return torch.randn(8, 14, 30)


@torch.no_grad()
@pytest.mark.parametrize("lstm_units", [[16], [16, 16], [16, 8]])
@pytest.mark.parametrize("fc_units", [4, None])
def test_forward(inputs, lstm_units, fc_units):
    input_channels = inputs.shape[1]
    output_channels = fc_units or lstm_units[-1]
    lstm = LstmExtractor(input_channels, lstm_units, fc_units)

    outputs = lstm(inputs)

    assert outputs.shape == torch.Size([8, output_channels])


@pytest.mark.parametrize("lstm_units", [[16, 16], [16, 8]])
def test_lstm_dropout(lstm_units):
    lstm = LstmExtractor(14, lstm_units, lstm_dropout=0.5)

    assert lstm.lstm_dropout == 0.5
    assert lstm._lstm_layers.dropout == 0.5


def test_fc_dropout():
    lstm = LstmExtractor(14, [16], 4, fc_dropout=0.5)

    dropout_layer = lstm._fc_layer[0]
    assert isinstance(dropout_layer, nn.Dropout)
    assert dropout_layer.p == 0.5
