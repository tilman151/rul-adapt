import re

import pytest
import torch
from torch import nn

from rul_adapt.model import CnnExtractor


@torch.no_grad()
@pytest.mark.parametrize("conv_filters", [[16, 16], [16, 8]])
@pytest.mark.parametrize("kernel_size", [2, 3, 4, 5, [3, 1]])
@pytest.mark.parametrize("fc_units", [4, None])
@pytest.mark.parametrize("padding", [True, False])
@pytest.mark.parametrize("batch_norm", [True, False])
def test_forward(inputs, conv_filters, kernel_size, fc_units, padding, batch_norm):
    input_channels = inputs.shape[1]
    cnn = CnnExtractor(
        input_channels,
        conv_filters,
        seq_len=30,
        kernel_size=kernel_size,
        padding=padding,
        fc_units=fc_units,
        batch_norm=batch_norm,
    )
    output_channels = fc_units or cnn._get_flat_dim()

    outputs = cnn(inputs)

    assert outputs.shape == torch.Size([8, output_channels])


@torch.no_grad()
@pytest.mark.parametrize("padding", [True, False])
def test_padding(inputs, padding):
    input_channels, seq_len, num_filters = inputs.shape[1], 30, 16
    cnn = CnnExtractor(input_channels, [num_filters], seq_len, padding=padding)

    outputs = cnn(inputs)

    exp_out_shape = seq_len * num_filters  # output is flattened
    assert (outputs.shape[-1] == exp_out_shape) == padding


def test_conv_layers(inputs):
    input_channels, seq_len, num_filters = inputs.shape[1], 30, 16
    cnn = CnnExtractor(input_channels, [num_filters], seq_len)

    def _check_conv_layer(module, module_name):
        assert len(module) == 2
        assert isinstance(module[0], nn.Conv1d)
        assert module[0].bias is not None

    _check_each_conv_layer(cnn, _check_conv_layer)


def test_batch_norm(inputs):
    input_channels, seq_len, num_filters = inputs.shape[1], 30, 16
    cnn = CnnExtractor(input_channels, [num_filters], seq_len, batch_norm=True)

    def _check_batch_norm(module, module_name):
        assert len(module) == 3
        assert isinstance(module[1], nn.BatchNorm1d)
        assert module[0].bias is None

    _check_each_conv_layer(cnn, _check_batch_norm)


def test_conv_dropout(inputs):
    input_channels, seq_len, num_filters = inputs.shape[1], 30, 16
    cnn = CnnExtractor(input_channels, [num_filters], seq_len, dropout=0.5)

    def _check_dropout(module, module_name):
        if module_name.endswith("conv_0"):
            assert len(module) == 2
            assert not isinstance(module[0], nn.Dropout1d)
        else:
            assert len(module) == 3
            assert isinstance(module[0], nn.Dropout1d)
            assert module[0].p == 0.5

    _check_each_conv_layer(cnn, _check_dropout)


@pytest.mark.parametrize("conv_act_func", [nn.LeakyReLU, "torch.nn.LeakyReLU"])
def test_conv_act_func(inputs, conv_act_func):
    input_channels, seq_len, num_filters = inputs.shape[1], 30, 16
    cnn = CnnExtractor(input_channels, [num_filters], seq_len, act_func=conv_act_func)

    def _check_act_func(module, module_name):
        assert isinstance(module[-1], nn.LeakyReLU)

    _check_each_conv_layer(cnn, _check_act_func)


def _check_each_conv_layer(cnn, check_func):
    conv_pattern = re.compile(r".*conv_\d$")
    checked_any = False
    for module_name, module in cnn.named_modules():
        if conv_pattern.match(module_name) is not None:
            check_func(module, module_name)
            checked_any = True

    if not checked_any:
        pytest.fail("Did not check any Conv layers.")


def test_fc_layer(inputs):
    input_channels, seq_len, num_filters = inputs.shape[1], 30, 16
    cnn = CnnExtractor(input_channels, [num_filters], seq_len, fc_units=10)

    def _check_fc_units(module):
        assert len(module) == 2
        assert module[0].out_features == 10

    _check_fc_layer(cnn, _check_fc_units)


def test_fc_dropout(inputs):
    input_channels, seq_len, num_filters = inputs.shape[1], 30, 16
    cnn = CnnExtractor(
        input_channels, [num_filters], seq_len, fc_units=10, fc_dropout=0.5
    )

    def _check_dropout(module):
        assert len(module) == 3
        assert module[0].p == 0.5

    _check_fc_layer(cnn, _check_dropout)


@pytest.mark.parametrize("fc_act_func", [nn.LeakyReLU, "torch.nn.LeakyReLU"])
def test_fc_act_func(inputs, fc_act_func):
    input_channels, seq_len, num_filters = inputs.shape[1], 30, 16
    cnn = CnnExtractor(
        input_channels, [num_filters], seq_len, fc_units=10, fc_act_func=fc_act_func
    )

    def _check_act_func(module):
        assert isinstance(module[-1], nn.LeakyReLU)

    _check_fc_layer(cnn, _check_act_func)


def _check_fc_layer(cnn, check_func):
    checked_any = False
    for module_name, module in cnn.named_modules():
        if module_name.endswith("fc"):
            check_func(module)
            checked_any = True

    if not checked_any:
        pytest.fail("Did not check any FC layers.")
