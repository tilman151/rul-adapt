from typing import Tuple, Any

import hydra
import omegaconf
import pytorch_lightning as pl
import rul_datasets

from rul_adapt.approach import DannApproach


def get_lstm_dann(
    source_fd: int, target_fd: int, **trainer_kwargs: Any
) -> Tuple[rul_datasets.DomainAdaptionDataModule, DannApproach, pl.Trainer]:
    """
    Construct an LSTM-DANN approach for CMAPSS with the original hyperparameters.

    Examples:
        ```pycon
        >>> import rul_adapt
        >>> dm, dann, trainer = rul_adapt.construct.get_lstm_dann(3, 1)
        >>> trainer.fit(dann, dm)
        >>> trainer.test(dann, dm)
        ```

    Args:
        source_fd: The source FD of CMAPSS.
        target_fd: The target FD of CMAPSS.
        trainer_kwargs: Overrides for the trainer class.
    Returns:
        dm: The data module for adaption of two CMAPSS sub-datasets.
        dann: The DANN approach with feature extractor, regressor and domain disc.
        trainer: The trainer object.
    """
    config = get_lstm_dann_config(source_fd, target_fd)
    dm, dann, trainer = lstm_dann_from_config(config, **trainer_kwargs)

    return dm, dann, trainer


def get_lstm_dann_config(source_fd: int, target_fd: int) -> omegaconf.DictConfig:
    """
    Get a configuration for the LSTM-DANN approach.

    The configuration can be modified and fed to [lstm_dann_from_config]
    [rul_adapt.construct.lstm_dann.lstm_dann_from_config] to create the approach.

    Args:
        source_fd: The source FD of CMAPSS.
        target_fd: The target FD of CMAPSS.
    Returns:
        The LSTM-DANN configuration.
    """
    _validate(source_fd, target_fd)
    with hydra.initialize("config", version_base="1.1"):
        config = hydra.compose("base", overrides=[f"+task={source_fd}-{target_fd}"])

    return config


def lstm_dann_from_config(
    config: omegaconf.DictConfig, **trainer_kwargs: Any
) -> Tuple[rul_datasets.DomainAdaptionDataModule, DannApproach, pl.Trainer]:
    """
    Construct a LSTM-DANN approach from a configuration.

    The configuration can be created by calling [get_lstm_dann_config]
    [rul_adapt.construct.lstm_dann.get_lstm_dann_config].

    Args:
        config: The LSTM-DANN configuration.
        trainer_kwargs: Overrides for the trainer class.
    Returns:
        dm: The data module for adaption of two CMAPSS sub-datasets.
        dann: The DANN approach with feature extractor, regressor and domain disc.
        trainer: The trainer object.
    """
    source = hydra.utils.instantiate(config.dm.source)
    target = source.get_compatible(**config.dm.target)
    dm = rul_datasets.DomainAdaptionDataModule(
        rul_datasets.RulDataModule(source, config.dm.batch_size),
        rul_datasets.RulDataModule(target, config.dm.batch_size),
    )

    feature_extractor = hydra.utils.instantiate(config.feature_extractor)
    regressor = hydra.utils.instantiate(config.regressor)
    domain_disc = hydra.utils.instantiate(config.domain_disc)

    dann = hydra.utils.instantiate(config.dann)
    dann.set_model(feature_extractor, regressor, domain_disc)

    trainer = hydra.utils.instantiate(config.trainer, **trainer_kwargs)

    return dm, dann, trainer


def _validate(source_fd: int, target_fd: int) -> None:
    if source_fd == target_fd:
        raise ValueError(
            f"No configuration for adapting from FD{source_fd:03} to itself."
        )
    elif 1 > source_fd or source_fd > 4:
        raise ValueError(f"CMAPSS has only FD001 to FD004 but no FD{source_fd:03}")
    elif 1 > target_fd or target_fd > 4:
        raise ValueError(f"CMAPSS has only FD001 to FD004 but no FD{target_fd:03}")
