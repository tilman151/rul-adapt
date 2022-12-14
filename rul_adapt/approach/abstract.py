"""A module for the abstract base class of all approaches."""
import warnings
from abc import ABCMeta
from typing import Any

import pytorch_lightning as pl
from torch import nn


class AdaptionApproach(pl.LightningModule, metaclass=ABCMeta):
    """
    This abstract class is the base of all adaption approaches.

    It defines that there needs to be a `feature_extractor`, a `regressor`. These
    members can be accessed via read-only properties. The `feature_extractor` and
    `regressor` are trainable neural networks.

    All child classes are supposed to implement their own constructors. The
    `feature_extractor` and `regressor` should explicitly not be arguments of the
    constructor and should be set by calling [set_model]
    [rul_adapt.approach.abstract.AdaptionApproach.set_model]. This way, the approach can
    be initialized with all hyperparameters first and afterwards supplied with the
    networks. This is useful for initializing the networks with pre-trained weights.
    """

    _feature_extractor: nn.Module
    _regressor: nn.Module

    def set_model(
        self,
        feature_extractor: nn.Module,
        regressor: nn.Module,
        *args: Any,
        **kwargs: Any
    ) -> None:
        """
        Set the feature extractor and regressor for this approach.

        Child classes can override this function to add additional models to an
        approach. The `args` and `kwargs` making this possible are ignored in this
        function.

        Args:
            feature_extractor: The feature extraction network.
            regressor: The RUL regression network.
        """
        self._feature_extractor = feature_extractor
        self._regressor = regressor
        if args:
            warnings.warn("Additional position args were supplied, which are ignored.")
        if kwargs:
            warnings.warn("Additional keyword args were supplied, which are ignored.")

    @property
    def feature_extractor(self) -> nn.Module:
        """The feature extraction network."""
        if hasattr(self, "_feature_extractor"):
            return self._feature_extractor
        else:
            raise RuntimeError("Feature extractor used before 'set_model' was called.")

    @property
    def regressor(self) -> nn.Module:
        """The RUL regression network."""
        if hasattr(self, "_regressor"):
            return self._regressor
        else:
            raise RuntimeError("Regressor used before 'set_model' was called.")
