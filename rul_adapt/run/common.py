import hydra.utils
import pytorch_lightning as pl


def should_be_pretrained(config):
    return "approach" in config["pretraining"]


def get_approach(config):
    return hydra.utils.instantiate(config["training"]["approach"])


def run_adaption(config, approach, dm):
    trainer = get_trainer(config)
    if is_wandb_logger(trainer.logger):
        trainer.logger.experiment.define_metric(
            "val/target/rmse/dataloader_idx_1", summary="best", goal="minimize"
        )
    trainer.fit(approach, dm)
    result = get_result(config, trainer, dm)
    if is_wandb_logger(trainer.logger):
        trainer.logger.experiment.finish()

    return result


def get_result(config, trainer, dm):
    if config["test"]:
        result = trainer.test(ckpt_path="best", datamodule=dm)
    else:
        result = trainer.checkpoint_callback.best_model_score.item()
    return result


def get_trainer(config):
    return hydra.utils.instantiate(config["training"]["trainer"])


def run_pretraining(config, dm):
    feature_extractor, regressor = get_models(config)
    approach_pretraining = hydra.utils.instantiate(config["pretraining"]["approach"])
    approach_pretraining.set_model(feature_extractor, regressor)
    trainer_pretraining = hydra.utils.instantiate(config["pretraining"]["trainer"])

    if is_wandb_logger(trainer_pretraining.logger):
        trainer_pretraining.logger.experiment.define_metric(
            "val/loss", summary="best", goal="minimize"
        )
    trainer_pretraining.fit(approach_pretraining, dm.source)
    if is_wandb_logger(trainer_pretraining.logger):
        trainer_pretraining.logger.experiment.finish()

    best_checkpoint = trainer_pretraining.checkpoint_callback.best_model_path
    best_pretrained = approach_pretraining.load_from_checkpoint(best_checkpoint)

    return best_pretrained


def get_models(config):
    feature_extractor = hydra.utils.instantiate(config["feature_extractor"])
    regressor = hydra.utils.instantiate(config["regressor"])

    return feature_extractor, regressor


def get_adaption_datamodule(config):
    source = hydra.utils.instantiate(config["source"])
    target = hydra.utils.instantiate(config["target"])
    dm = hydra.utils.instantiate(config["dm"], source, target)

    return dm


def is_wandb_logger(logger):
    return isinstance(logger, pl.loggers.WandbLogger)
