import pytest
import torch
from torch import nn

from rul_adapt.model.head import FullyConnectedHead


@pytest.mark.parametrize(
    "units", [[1], [32, 1], [32, 2], pytest.param([], marks=pytest.mark.xfail)]
)
def test_forward(inputs, units):
    inputs = torch.flatten(inputs, start_dim=1)
    head = FullyConnectedHead(inputs.shape[1], units)

    outputs = head(inputs)

    assert outputs.shape == torch.Size([8, units[-1]])


@pytest.mark.parametrize("units", [[1], [32, 1]])
@pytest.mark.parametrize("act_func", [nn.ReLU, nn.Sigmoid])
def test_act_func(units, act_func):
    head = FullyConnectedHead(14, units, act_func=act_func)

    for m in head._layers:
        assert isinstance(m[-1], act_func)


@pytest.mark.parametrize("act_func_on_last_layer", [True, False])
def test_act_func_on_last_layer(act_func_on_last_layer):
    head = FullyConnectedHead(
        14, [10, 1], act_func_on_last_layer=act_func_on_last_layer
    )

    assert isinstance(head._layers[-1], nn.Sequential)
    if act_func_on_last_layer:
        assert isinstance(head._layers[-1][-1], head.act_func)
    else:
        assert isinstance(head._layers[-1][-1], nn.Linear)


@pytest.mark.parametrize("dropout", [0.0, 0.1])
def test_dropout(dropout):
    head = FullyConnectedHead(14, [10, 1], dropout)

    for module in head._layers:
        if dropout:
            assert isinstance(module[0], nn.Dropout)
            assert module[0].p == dropout
        else:
            assert not isinstance(module[0], nn.Dropout)
