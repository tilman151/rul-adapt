from typing import Any, Dict

import hydra.utils
import torch
from torch.utils.data import ConcatDataset, DataLoader

import rul_adapt
from rul_adapt.run import common


def pseudo_labels(config: Dict[str, Any]):
    dm = common.get_adaption_datamodule(config)
    best_pretrained = common.run_pretraining(config, dm)
    approach = common.get_approach(config)
    approach.set_model(best_pretrained.feature_extractor, best_pretrained.regressor)

    combined_dl = _get_adaption_dataloader(approach, dm)
    trainer = common.get_trainer(config)
    if common.is_wandb_logger(trainer.logger):
        trainer.logger.experiment.define_metric(
            "val/loss", summary="best", goal="minimize"
        )
    trainer.fit(
        approach,
        train_dataloaders=combined_dl,
        val_dataloaders=dm.target.val_dataloader(),
    )
    result = common.get_result(config, trainer, dm)
    if common.is_wandb_logger(trainer.logger):
        trainer.logger.experiment.finish()

    return result


def _get_adaption_dataloader(approach, dm):
    pseudo_rul = rul_adapt.approach.generate_pseudo_labels(
        dm.target, approach, dm.inductive
    )
    pseudo_rul = [min(dm.target.reader.max_rul, max(0.0, pr)) for pr in pseudo_rul]
    rul_adapt.approach.patch_pseudo_labels(dm.target, pseudo_rul, dm.inductive)

    source_data = dm.source.to_dataset("dev")
    target_data = dm.target.to_dataset("test" if dm.inductive else "dev", alias="dev")
    combined_data = ConcatDataset([source_data, target_data])
    combined_dl = DataLoader(combined_data, dm.source.batch_size, shuffle=True)

    return combined_dl
