"""This module contains the necessary neural networks to build a RUL estimator.

In general, a RUL estimator network consists of two parts: a feature extractor and a
regression head. The feature extractor is a network that transforms the input feature
windows into a latent feature vector. The regression head maps these latent features
to a scalar RUL value. The feature extractors can be found in the [cnn]
[rul_adapt.model.cnn] and [rnn][rul_adapt.model.rnn] modules. The regression head in
the [head][rul_adapt.model.head] module.

Some domain adaption approaches use a domain discriminator. The networks in the
[head][rul_adapt.model.head] module can be used to construct them, too.

Examples:
    >>> import torch
    >>> from rul_adapt import model
    >>> feature_extractor = model.CnnExtractor(14,[32, 16],30,fc_units=8)
    >>> regressor = model.FullyConnectedHead(8, [4, 1])
    >>> latent_features = feature_extractor(torch.randn(10, 14, 30))
    >>> latent_features.shape
    torch.Size([10, 8])
    >>> rul = regressor(latent_features)
    >>> rul.shape
    torch.Size([10, 1])
"""

from .cnn import CnnExtractor
from .head import FullyConnectedHead
from .rnn import LstmExtractor, GruExtractor
from .wrapper import ActivationDropoutWrapper
