from typing import Dict, Any

from rul_adapt.run import common


def no_adaption(config: Dict[str, Any]):
    dm = common.get_adaption_datamodule(config)
    approach = common.get_approach(config)
    approach.set_model(*common.get_models(config))
    trainer = common.get_trainer(config)
    if common.is_wandb_logger(trainer.logger):
        trainer.logger.experiment.define_metric(
            "val/loss", summary="best", goal="minimize"
        )
        trainer.logger.experiment.config["approach"] = "NoAdaptionApproach"
        trainer.logger.log_hyperparams(dm.hparams)  # log manually as dm isn't used
    dm.prepare_data()  # needs to be called because only dataloaders are used
    dm.setup()
    trainer.fit(
        approach,
        train_dataloaders=dm.source.train_dataloader(),
        val_dataloaders=dm.target.val_dataloader(),
    )
    result = common.get_result(config, trainer, dm)
    if common.is_wandb_logger(trainer.logger):
        trainer.logger.experiment.finish()

    return result
