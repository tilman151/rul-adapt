import numpy.testing as npt
import pytest
import torch
from torch import nn

from rul_adapt.model.rnn import LstmExtractor, GruExtractor, _Rnn


@torch.no_grad()
@pytest.mark.parametrize("lstm_units", [[16], [16, 16], [16, 8]])
@pytest.mark.parametrize("fc_units", [4, None])
@pytest.mark.parametrize("bidirectional", [True, False])
def test_lstm_forward(inputs, lstm_units, fc_units, bidirectional):
    input_channels = inputs.shape[1]
    output_channels = fc_units or (lstm_units[-1] * (2 if bidirectional else 1))
    lstm = LstmExtractor(
        input_channels, lstm_units, fc_units, bidirectional=bidirectional
    )

    outputs = lstm(inputs)

    assert outputs.shape == torch.Size([8, output_channels])


@pytest.mark.parametrize("lstm_units", [[16, 16], [16, 8]])
def test_lstm_dropout(lstm_units):
    lstm = LstmExtractor(14, lstm_units, lstm_dropout=0.5)

    assert lstm.lstm_dropout == 0.5
    assert lstm._lstm_layers.dropout == 0.5


@torch.no_grad()
@pytest.mark.parametrize("rnn_func", [nn.LSTM, nn.GRU])
def test_rnn_dropout_equivalent(rnn_func):
    inputs = torch.randn(30, 10, 14)
    with torch.random.fork_rng():
        default_lstm = rnn_func(14, 16, 3, dropout=0.5)
        outputs_default, states_default = default_lstm(inputs)
    with torch.random.fork_rng():
        custom_lstm = _Rnn(rnn_func, 14, [16, 16, 16], dropout=0.5, bidirectional=False)
        outputs_custom, states_custom = custom_lstm(inputs)

    npt.assert_almost_equal(outputs_custom.numpy(), outputs_default.numpy())
    if isinstance(states_custom, tuple):
        for custom, default in zip(states_custom, states_default):
            npt.assert_almost_equal(custom.numpy(), default[2:].numpy())
    else:
        npt.assert_almost_equal(states_custom.numpy(), states_default[2:].numpy())


def test_fc_dropout():
    lstm = LstmExtractor(14, [16], 4, fc_dropout=0.5)

    dropout_layer = lstm._fc_layer[0]
    assert isinstance(dropout_layer, nn.Dropout)
    assert dropout_layer.p == 0.5


@pytest.mark.parametrize("lstm_units", [[16, 16], [16, 8]])
def test_lstm_bidirectional(lstm_units):
    lstm = LstmExtractor(14, lstm_units, bidirectional=True)

    assert lstm.bidirectional
    lstm_layer = lstm._lstm_layers
    assert lstm_layer.bidirectional


@torch.no_grad()
@pytest.mark.parametrize("gru_units", [[16], [16, 16], [16, 8]])
@pytest.mark.parametrize("fc_units", [[4], [4, 4]])
@pytest.mark.parametrize("bidirectional", [True, False])
def test_gru_forward(inputs, gru_units, fc_units, bidirectional):
    input_channels = inputs.shape[1]
    output_channels = gru_units[-1] * (2 if bidirectional else 1)
    gru = GruExtractor(input_channels, fc_units, gru_units, bidirectional=bidirectional)

    outputs = gru(inputs)

    assert outputs.shape == torch.Size([8, output_channels])


@pytest.mark.parametrize("gru_units", [[16, 16], [16, 8]])
def test_gru_dropout(gru_units):
    gru = GruExtractor(14, [4], gru_units, gru_dropout=0.5)

    assert gru.gru_dropout == 0.5
    assert gru._gru_layers.dropout == 0.5


@pytest.mark.parametrize("gru_units", [[16, 16], [16, 8]])
def test_gru_bidirectional(gru_units):
    gru = GruExtractor(14, [4], gru_units, bidirectional=True)

    assert gru.bidirectional
    gru_layer = gru._gru_layers
    assert gru_layer.bidirectional
