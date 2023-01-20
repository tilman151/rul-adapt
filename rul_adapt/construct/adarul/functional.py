from typing import Tuple, Optional, Dict, Any

import hydra
import omegaconf
import rul_datasets
import pytorch_lightning as pl
from torch import nn

from rul_adapt.approach import AdaRulApproachPretraining, AdaRulApproach


def get_adarul(
    source_fd: int,
    target_fd: int,
    pre_trainer_kwargs: Optional[Dict[str, Any]] = None,
    trainer_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[
    Tuple[rul_datasets.RulDataModule, AdaRulApproachPretraining, pl.Trainer],
    Tuple[
        rul_datasets.DomainAdaptionDataModule,
        nn.Module,
        AdaRulApproach,
        pl.Trainer,
    ],
]:
    config = get_adarul_config(source_fd, target_fd)
    adarul = adarul_from_config(config, pre_trainer_kwargs, trainer_kwargs)

    return adarul


def get_adarul_config(source_fd: int, target_fd: int) -> omegaconf.DictConfig:
    _validate(source_fd, target_fd)
    with hydra.initialize("config", version_base="1.1"):
        config = hydra.compose(
            "base",
            overrides=[f"dm.source.fd={source_fd}", f"dm.target.fd={target_fd}"],
        )

    return config


def adarul_from_config(
    config: omegaconf.DictConfig,
    pre_trainer_kwargs: Optional[Dict[str, Any]] = None,
    trainer_kwargs: Optional[Dict[str, Any]] = None,
):
    source = hydra.utils.instantiate(config.dm.source)
    target = source.get_compatible(**config.dm.target)
    dm_pre = rul_datasets.RulDataModule(source, config.dm.batch_size)
    dm = rul_datasets.DomainAdaptionDataModule(
        dm_pre, rul_datasets.RulDataModule(target, config.dm.batch_size)
    )

    feature_extractor = hydra.utils.instantiate(config.feature_extractor)
    regressor = hydra.utils.instantiate(config.regressor)
    domain_disc = hydra.utils.instantiate(config.domain_disc)

    approach_pre = hydra.utils.instantiate(config.adarul_pre)
    approach_pre.set_model(feature_extractor, regressor)

    approach = hydra.utils.instantiate(config.adarul)

    pre_trainer_kwargs = pre_trainer_kwargs or {}
    trainer_kwargs = trainer_kwargs or {}
    trainer_pre = hydra.utils.instantiate(config.trainer_pre, **pre_trainer_kwargs)
    trainer = hydra.utils.instantiate(config.trainer, **trainer_kwargs)

    return (dm_pre, approach_pre, trainer_pre), (dm, approach, domain_disc, trainer)


def _validate(source_fd: int, target_fd: int):
    if 1 > source_fd or source_fd > 4:
        raise ValueError(f"CMAPSS has only FD001 to FD004 but no FD{source_fd:03}")
    elif 1 > target_fd or target_fd > 4:
        raise ValueError(f"CMAPSS has only FD001 to FD004 but no FD{target_fd:03}")
