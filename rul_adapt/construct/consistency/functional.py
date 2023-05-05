from typing import Literal, Tuple, Any, Dict, Optional

import hydra
import omegaconf
import rul_datasets
import pytorch_lightning as pl
from torch import nn

from rul_adapt.approach import ConsistencyApproach, SupervisedApproach


def get_consistency_dann(
    dataset: Literal["cmapss", "xjtu-sy"],
    source_fd: int,
    target_fd: int,
    pre_trainer_kwargs: Optional[Dict[str, Any]] = None,
    trainer_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[
    Tuple[rul_datasets.RulDataModule, SupervisedApproach, pl.Trainer],
    Tuple[
        rul_datasets.DomainAdaptionDataModule,
        nn.Module,
        ConsistencyApproach,
        pl.Trainer,
    ],
]:
    """
    Construct a Consistency DANN approach with the original hyperparameters.

    Examples:
        ```pycon
        >>> import rul_adapt
        >>> pre, main = rul_adapt.construct.get_consistency_dann("cmapss", 3, 1)
        >>> pre_dm, pre_approach, pre_trainer = pre
        >>> dm, approach, domain_disc, trainer = main
        ```

    Args:
        dataset: The name of the dataset, either `cmapss` or `xjtu-sy`.
        source_fd: The source FD of CMAPSS.
        target_fd: The target FD of CMAPSS.
        pre_trainer_kwargs: Overrides for the pre-training trainer class.
        trainer_kwargs: Overrides for the main trainer class.
    Returns:
        pre: The data module, approach and trainer for the pre-training stage
        main: The data module, approach, domain discriminator and trainer for
              the main stage
    """
    config = get_consistency_dann_config(dataset, source_fd, target_fd)
    consistency_dann = consistency_dann_from_config(
        config, pre_trainer_kwargs, trainer_kwargs
    )

    return consistency_dann


def get_consistency_dann_config(
    dataset: Literal["cmapss", "xjtu-sy"], source_fd: int, target_fd: int
) -> omegaconf.DictConfig:
    """
    Get a configuration for the Consistency DANN approach.

    The configuration can be modified and fed to [consistency_dann_from_config]
    [rul_adapt.construct.consistency.consistency_dann_from_config] to create the
    approach.

    Args:
        dataset: The name of the dataset, either `cmapss` or `xjtu-sy`.
        source_fd: The source FD of CMAPSS.
        target_fd: The target FD of CMAPSS.
    Returns:
        The Consistency DANN configuration.
    """
    _validate(dataset, source_fd, target_fd)
    with hydra.initialize("config", version_base="1.1"):
        config = hydra.compose(
            "base",
            overrides=[
                f"+dataset={dataset}",
                f"dm.source.fd={source_fd}",
                f"dm.target.fd={target_fd}",
            ],
        )

    return config


def consistency_dann_from_config(
    config: omegaconf.DictConfig,
    pre_trainer_kwargs: Optional[Dict[str, Any]] = None,
    trainer_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[
    Tuple[rul_datasets.RulDataModule, SupervisedApproach, pl.Trainer],
    Tuple[
        rul_datasets.DomainAdaptionDataModule,
        nn.Module,
        ConsistencyApproach,
        pl.Trainer,
    ],
]:
    """
    Construct a Consistency DANN approach from a configuration.
    The configuration can be created by calling [get_consistency_dann_config]
    [rul_adapt.construct.consistency.get_consistency_dann_config].

    Args:
        config: The Consistency DANN config.
        pre_trainer_kwargs: Overrides for the pre-training trainer class.
        trainer_kwargs: Overrides for the main trainer class.
    Returns:
        pre: The data module, approach and trainer for the pre-training stage
        main: The data module, approach, domain discriminator and trainer for
              the main stage
    """
    source = hydra.utils.instantiate(config.dm.source)
    target = source.get_compatible(**config.dm.target)
    kwargs = hydra.utils.instantiate(config.dm.kwargs)
    dm_pre = rul_datasets.RulDataModule(source, **kwargs)
    dm = rul_datasets.DomainAdaptionDataModule(
        dm_pre, rul_datasets.RulDataModule(target, **kwargs)
    )

    feature_extractor = hydra.utils.instantiate(config.feature_extractor)
    regressor = hydra.utils.instantiate(config.regressor)
    domain_disc = hydra.utils.instantiate(config.domain_disc)

    approach_pre = hydra.utils.instantiate(config.consistency_pre)
    approach_pre.set_model(feature_extractor, regressor)

    approach = hydra.utils.instantiate(config.consistency)

    pre_trainer_kwargs = pre_trainer_kwargs or {}
    trainer_kwargs = trainer_kwargs or {}
    trainer_pre = hydra.utils.instantiate(config.trainer_pre, **pre_trainer_kwargs)
    trainer = hydra.utils.instantiate(config.trainer, **trainer_kwargs)

    return (dm_pre, approach_pre, trainer_pre), (dm, approach, domain_disc, trainer)


def _validate(
    dataset: Literal["cmapss", "xjtu-sy"], source_fd: int, target_fd: int
) -> None:
    if dataset not in ["cmapss", "xjtu-sy"]:
        raise ValueError(f"No configuration for '{dataset}'.")
    elif source_fd == target_fd:
        raise ValueError(
            f"No configuration for adapting from FD{source_fd:03} to itself."
        )
    elif dataset == "cmapss":
        _validate_cmapss(source_fd, target_fd)
    elif dataset == "xjtu-sy":
        _validate_xjtu_sy(source_fd, target_fd)


def _validate_cmapss(source_fd: int, target_fd: int):
    if 1 > source_fd or source_fd > 4:
        raise ValueError(f"CMAPSS has only FD001 to FD004 but no FD{source_fd:03}")
    elif 1 > target_fd or target_fd > 4:
        raise ValueError(f"CMAPSS has only FD001 to FD004 but no FD{target_fd:03}")


def _validate_xjtu_sy(source_fd: int, target_fd: int):
    if 1 > source_fd or source_fd > 3:
        raise ValueError(f"XJTU-SY has only FD001 to FD003 but no FD{source_fd:03}")
    elif 1 > target_fd or target_fd > 3:
        raise ValueError(f"XJTU-SY has only FD001 to FD003 but no FD{target_fd:03}")
