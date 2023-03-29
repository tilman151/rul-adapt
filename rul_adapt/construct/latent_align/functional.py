from typing import Literal, Tuple, Any, Optional

import hydra
import omegaconf
import pytorch_lightning as pl
import rul_datasets

from rul_adapt.approach import LatentAlignApproach


def get_latent_align(
    dataset: Literal["cmapss", "xjtu-sy"],
    source_fd: int,
    target_fd: int,
    xjtu_sy_subtask: Optional[int] = None,
    **trainer_kwargs: Any,
) -> Tuple[rul_datasets.LatentAlignDataModule, LatentAlignApproach, pl.Trainer]:
    config = get_latent_align_config(dataset, source_fd, target_fd, xjtu_sy_subtask)
    dm, latent_align, trainer = latent_align_from_config(config, **trainer_kwargs)

    return dm, latent_align, trainer


def get_latent_align_config(
    dataset: Literal["cmapss", "xjtu-sy"],
    source_fd: int,
    target_fd: int,
    xjtu_sy_subtask: Optional[int] = None,
) -> omegaconf.DictConfig:
    _validate(dataset, source_fd, target_fd, xjtu_sy_subtask)
    overrides = [
        f"+dataset={dataset}",
        f"dm.source.fd={source_fd}",
        f"dm.target.fd={target_fd}",
    ]
    if dataset == "xjtu-sy":
        overrides.append(f"+subtask=SUB00{xjtu_sy_subtask}")
    else:
        overrides.append(f"+split_steps=FD00{target_fd}")
    with hydra.initialize("config", version_base="1.1"):
        config = hydra.compose("base", overrides)

    return config


def latent_align_from_config(
    config: omegaconf.DictConfig, **trainer_kwargs: Any
) -> Tuple[rul_datasets.LatentAlignDataModule, LatentAlignApproach, pl.Trainer]:
    source = hydra.utils.instantiate(config.dm.source)
    target = hydra.utils.instantiate(config.dm.target)
    kwargs = hydra.utils.instantiate(config.dm.kwargs)
    dm = rul_datasets.LatentAlignDataModule(
        rul_datasets.RulDataModule(source, **kwargs),
        rul_datasets.RulDataModule(target, **kwargs),
        **config.dm.adaption_kwargs,
    )

    feature_extractor = hydra.utils.instantiate(config.feature_extractor)
    regressor = hydra.utils.instantiate(config.regressor)

    approach = hydra.utils.instantiate(config.latent_align)
    approach.set_model(feature_extractor, regressor)

    trainer = hydra.utils.instantiate(config.trainer, **trainer_kwargs)

    return dm, approach, trainer


def _validate(
    dataset: Literal["cmapss", "xjtu-sy"],
    source_fd: int,
    target_fd: int,
    xjtu_sy_subtask: Optional[int],
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
        _validate_xjtu_sy(source_fd, target_fd, xjtu_sy_subtask)


def _validate_cmapss(source_fd: int, target_fd: int):
    if 1 > source_fd or source_fd > 4:
        raise ValueError(f"CMAPSS has only FD001 to FD004 but no FD{source_fd:03}")
    elif 1 > target_fd or target_fd > 4:
        raise ValueError(f"CMAPSS has only FD001 to FD004 but no FD{target_fd:03}")


def _validate_xjtu_sy(source_fd: int, target_fd: int, subtask: int):
    if 1 > source_fd or source_fd > 2:
        raise ValueError(
            "Only FD001 and FD002 of XJTU-SY are used in "
            f"this approach but not FD{source_fd:03}."
        )
    elif 1 > target_fd or target_fd > 2:
        raise ValueError(
            "Only FD001 and FD002 of XJTU-SY are used in "
            f"this approach but not FD{target_fd:03}."
        )
    elif subtask is None:
        raise ValueError("XJTU-SY requires a subtask of 1 or 2.")
    elif 1 > subtask or subtask > 2:
        raise ValueError(
            f"XJTU-SY has only subtasks 1 and 2 but not subtask {subtask}."
        )
