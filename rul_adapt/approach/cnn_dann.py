from functools import partial
from typing import cast

from torch import nn

from rul_adapt.model import CnnExtractor
from rul_adapt.model.wrapper import DropoutPrefix


def init_weights(feature_extractor: CnnExtractor, regressor: DropoutPrefix) -> None:
    """
    Initialize the weights of the feature extractor and regressor in-place.

    For the weight matrices the Xavier uniform initialization is used. The biases are
    initialized to zero. This function works only for the networks returned by a call to
    [rul_adapt.construct.get_cnn_dann][].

    Args:
        feature_extractor: The feature extractor network to be initialized.
        regressor: The regressor network to be initialized.
    """
    feature_extractor.apply(_weight_init_fe)
    layer_count = 0
    for m in regressor.modules():
        if layer_count < len(cast(list, regressor.wrapped.units)) - 1:
            m.apply(_weight_init_fc_tanh)
            layer_count += 1
        else:
            m.apply(_weight_init_fc_linear)


def _weight_init_fe(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("tanh"))
        nn.init.zeros_(m.bias)


def _weight_init_fc(m, gain):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=gain)
        nn.init.zeros_(m.bias)


_weight_init_fc_tanh = partial(_weight_init_fc, gain=nn.init.calculate_gain("tanh"))
_weight_init_fc_linear = partial(_weight_init_fc, gain=nn.init.calculate_gain("linear"))
