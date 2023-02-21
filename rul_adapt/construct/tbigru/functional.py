from typing import Tuple, Any

import hydra
import omegaconf
import pytorch_lightning as pl
import rul_datasets

from rul_adapt.approach import TBiGruApproach


def get_tbigru(
    source_fd: int, target_fd: int, **trainer_kwargs: Any
) -> Tuple[rul_datasets.DomainAdaptionDataModule, TBiGruApproach, pl.Trainer]:
    config = get_tbigru_config(source_fd, target_fd)
    dm, dann, trainer = tbigru_from_config(config, **trainer_kwargs)

    return dm, dann, trainer


def get_tbigru_config(source_fd: int, target_fd: int) -> omegaconf.DictConfig:
    _validate(source_fd, target_fd)
    with hydra.initialize("config", version_base="1.1"):
        config = hydra.compose("base", overrides=[f"+task={source_fd}-{target_fd}"])

    return config


def tbigru_from_config(
    config: omegaconf.DictConfig, **trainer_kwargs: Any
) -> Tuple[rul_datasets.DomainAdaptionDataModule, TBiGruApproach, pl.Trainer]:
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
