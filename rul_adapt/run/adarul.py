from typing import Any, Dict

import hydra.utils
import pytorch_lightning as pl


def adarul(config: Dict[str, Any]):
    source = hydra.utils.instantiate(config["source"])
    target = hydra.utils.instantiate(config["target"])

    feature_extractor = hydra.utils.instantiate(config["feature_extractor"])
    regressor = hydra.utils.instantiate(config["regressor"])
    approach_pretraining = hydra.utils.instantiate(config["pretraining"]["approach"])
    approach_pretraining.set_model(feature_extractor, regressor)

    trainer_pretraining = hydra.utils.instantiate(config["pretraining"]["trainer"])
    if _is_wandb_logger(trainer_pretraining.logger):
        trainer_pretraining.logger.experiment.define_metric(
            "val/loss", summary="best", goal="minimize"
        )
    trainer_pretraining.fit(approach_pretraining, source)
    if _is_wandb_logger(trainer_pretraining.logger):
        trainer_pretraining.logger.experiment.finish()

    best_checkpoint = trainer_pretraining.checkpoint_callback.best_model_path
    best_pretrained = approach_pretraining.load_from_checkpoint(best_checkpoint)

    dm = hydra.utils.instantiate(config["dm"], source, target)
    domain_disc = hydra.utils.instantiate(config["domain_disc"])
    approach = hydra.utils.instantiate(config["training"]["approach"])
    approach.set_model(
        best_pretrained.feature_extractor, best_pretrained.regressor, domain_disc
    )

    trainer = hydra.utils.instantiate(config["training"]["trainer"])
    if _is_wandb_logger(trainer.logger):
        trainer.logger.experiment.define_metric(
            "val/target/rmse/dataloader_idx_1", summary="best", goal="minimize"
        )
    trainer.fit(approach, dm)
    if config["test"]:
        result = trainer.test(ckpt_path="best", datamodule=dm)
    else:
        result = trainer.checkpoint_callback.best_model_score.item()
    if _is_wandb_logger(trainer.logger):
        trainer.logger.experiment.finish()

    return result


def _is_wandb_logger(logger):
    return isinstance(logger, pl.loggers.WandbLogger)
