import torch
from torch import nn

import rul_adapt
from rul_adapt.approach.cnn_dann import init_weights


def test_init_weights():
    fe = rul_adapt.model.CnnExtractor(14, [10, 10, 1], 30, 10)
    reg = rul_adapt.model.FullyConnectedHead(30, [32, 1])
    wrapped_reg = rul_adapt.model.wrapper.DropoutPrefix(reg, 0.5)

    rul_adapt.approach.cnn_dann.init_weights(fe, wrapped_reg)

    for m in fe.children():
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
            assert torch.all(m.bias == 0)

    for m in wrapped_reg.children():
        if isinstance(m, nn.Linear):
            assert torch.all(m.bias == 0)
