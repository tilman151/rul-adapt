"""A module for the abstract base class of all approaches."""
import inspect
import warnings
from abc import ABCMeta
from typing import Any, Dict, List, Set

import hydra.utils
import pytorch_lightning as pl
from torch import nn

EXCLUDED_ARGS = ["self", "device", "dtype"]  # init args ignored when checkpointing


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
    be initialized with all hyperparameters first and afterward supplied with the
    networks. This is useful for initializing the networks with pre-trained weights.

    Because models are constructed outside the approach, the default checkpointing
    mechanism of PyTorch Lightning fails to load checkpoints of AdaptionApproaches.
    We extended the checkpointing mechanism by implementing the `on_save_checkpoint`
    and `on_load_checkpoint` callbacks to make it work. If a subclass uses an
    additional model, besides feature extractor and regressor, that is not
    initialized in the constructor, the subclass needs to implement the
    `CHECKPOINT_MODELS` class variable. This variable is a list of model names to be
    included in the checkpoint. For example, if your approach has an additional model
    `self._domain_disc`, the `CHECKPOINT_MODELS` variable should be set to
    `['_domain_disc']`. Otherwise, loading a checkpoint of this approach will fail.
    """

    CHECKPOINT_MODELS: List[str] = []

    _feature_extractor: nn.Module
    _regressor: nn.Module

    _hparams_initial: Dict[str, Any]
    _logged_models: Dict[str, Set[str]]

    def set_model(
        self,
        feature_extractor: nn.Module,
        regressor: nn.Module,
        *args: Any,
        **kwargs: Any,
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

        self.log_model_hyperparameters("feature_extractor", "regressor")

    def log_model_hyperparameters(self, *model_names: str) -> None:
        hparams_initial = self.hparams_initial
        if not hasattr(self, "_logged_models"):
            self._logged_models = {}
            hparams_initial["model"] = {}

        for model_name in model_names:
            model_hparams = self._get_model_hparams(model_name)
            hparams_initial["model"].update(model_hparams)
            self._logged_models[model_name] = set(model_hparams.keys())

        self._hparams_initial = hparams_initial
        self._set_hparams(self._hparams_initial)

    def _get_model_hparams(self, model_name):
        prefix = model_name.lstrip("_")
        model = getattr(self, model_name)
        hparams = {prefix: {"type": type(model).__name__}}
        init_args = _get_init_args(model, "logging model hyperparameters")
        hparams[prefix].update(init_args)

        return hparams

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

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        del checkpoint["hyper_parameters"]["model"]
        checkpoint["logged_models"] = list(self._logged_models)
        to_checkpoint = ["_feature_extractor", "_regressor"] + self.CHECKPOINT_MODELS
        configs = {m: _get_hydra_config(getattr(self, m)) for m in to_checkpoint}
        checkpoint["model_configs"] = configs

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        for name, config in checkpoint["model_configs"].items():
            setattr(self, name, hydra.utils.instantiate(config))
        self.log_model_hyperparameters(*checkpoint["logged_models"])


def _get_hydra_config(model: nn.Module) -> Dict[str, Any]:
    model_type = type(model)
    class_name = f"{model_type.__module__}.{model_type.__qualname__}"
    config = {"_target_": class_name, **_get_init_args(model)}

    return config


def _get_init_args(
    obj: nn.Module, activity: str = "writing a checkpoint"
) -> Dict[str, Any]:
    if isinstance(obj, nn.ModuleList):
        # workaround because ModuleList's init arg is shadowed by a property
        init_args = {"modules": [_get_hydra_config(m) for m in obj]}
    elif isinstance(obj, nn.Sequential):
        # workaround because Sequential expects positional args only
        init_args = {"_args_": [_get_hydra_config(m) for m in obj]}
    else:
        signature = inspect.signature(type(obj).__init__)
        init_args = dict()
        arg_names = filter(lambda a: a not in EXCLUDED_ARGS, signature.parameters)
        for arg_name in arg_names:
            _check_has_attr(obj, arg_name, activity)
            arg = getattr(obj, arg_name)
            if isinstance(arg, nn.Module):
                arg = _get_hydra_config(arg)
            init_args[arg_name] = arg

    return init_args


def _check_has_attr(obj: Any, param: str, activity: str) -> None:
    if not hasattr(obj, param):
        raise RuntimeError(
            f"Error while {activity}. "
            f"The nn.Module of type '{type(obj)}' has an initialization parameter "
            f"named '{param}' which is not saved as a member variable, i.e. "
            f"'self.{param}'. Therefore, we cannot retrieve the value of "
            f"'{param}' the object was initialized with."
        )
