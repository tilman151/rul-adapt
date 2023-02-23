from typing import Tuple, Any

import hydra
import omegaconf
import pytorch_lightning as pl
import rul_datasets

from rul_adapt.approach import MmdApproach


def get_tbigru(
    source_fd: int, target_fd: int, **trainer_kwargs: Any
) -> Tuple[rul_datasets.DomainAdaptionDataModule, MmdApproach, pl.Trainer]:
    """
    Construct a TBiGRU approach for FEMTO with the original hyperparameters.

    Examples:
        ```pycon
        >>> import rul_adapt
        >>> dm, tbigru, trainer = rul_adapt.construct.get_tbigru(3, 1)
        >>> trainer.fit(tbigru, dm)
        >>> trainer.test(tbigru, dm)
        ```

    Args:
        source_fd: The source FD of FEMTO.
        target_fd: The target FD of FEMTO.
        trainer_kwargs: Overrides for the trainer class.
    Returns:
        dm: The data module for adaption of two FEMTO sub-datasets.
        dann: The TBiGRU approach with feature extractor and regressor.
        trainer: The trainer object.
    """
    config = get_tbigru_config(source_fd, target_fd)
    dm, dann, trainer = tbigru_from_config(config, **trainer_kwargs)

    return dm, dann, trainer


def get_tbigru_config(source_fd: int, target_fd: int) -> omegaconf.DictConfig:
    """
    Get a configuration for the TBiGRU approach.

    The configuration can be modified and fed to [tbigru_from_config]
    [rul_adapt.construct.tbigru.tbigru_from_config] to create the approach.

    Args:
        source_fd: The source FD of FEMTO.
        target_fd: The target FD of FEMTO.
    Returns:
        The TBiGRU configuration.
    """
    _validate(source_fd, target_fd)
    with hydra.initialize("config", version_base="1.1"):
        config = hydra.compose("base", overrides=[f"+task={source_fd}-{target_fd}"])

    return config


def tbigru_from_config(
    config: omegaconf.DictConfig, **trainer_kwargs: Any
) -> Tuple[rul_datasets.DomainAdaptionDataModule, MmdApproach, pl.Trainer]:
    """
    Construct a TBiGRU approach from a configuration.
    The configuration can be created by calling [get_tbigru_config]
    [rul_adapt.construct.tbigru.get_tbigru_config].

    Args:
        config: The TBiGRU configuration.
        trainer_kwargs: Overrides for the trainer class.
    Returns:
        dm: The data module for adaption of two FEMTO sub-datasets.
        dann: The TBiGRU approach with feature extractor and regressor.
        trainer: The trainer object.
    """
    source = hydra.utils.instantiate(config.dm.source)
    target = hydra.utils.instantiate(config.dm.target)

    source.prepare_data()  # needed in case this is the first use of FEMTO
    target.prepare_data()
    extractor = hydra.utils.instantiate(config.dm.feature_extractor)
    extractor.fit(source.load_split("dev")[0] + target.load_split("dev")[0])

    dm = rul_datasets.DomainAdaptionDataModule(
        rul_datasets.RulDataModule(
            source, config.dm.batch_size, extractor, config.dm.window_size
        ),
        rul_datasets.RulDataModule(
            target, config.dm.batch_size, extractor, config.dm.window_size
        ),
    )

    feature_extractor = hydra.utils.instantiate(config.feature_extractor)
    regressor = hydra.utils.instantiate(config.regressor)

    tbigru = hydra.utils.instantiate(config.tbigru)
    tbigru.set_model(feature_extractor, regressor)

    trainer = hydra.utils.instantiate(config.trainer, **trainer_kwargs)

    return dm, tbigru, trainer


def _validate(source_fd: int, target_fd: int) -> None:
    if source_fd == target_fd:
        raise ValueError(
            f"No configuration for adapting from FD{source_fd:03} to itself."
        )
    elif 1 > source_fd or source_fd > 3:
        raise ValueError(f"FEMTO has only FD001 to FD003 but no FD{source_fd:03}")
    elif 1 > target_fd or target_fd > 3:
        raise ValueError(f"FEMTO has only FD001 to FD003 but no FD{target_fd:03}")
