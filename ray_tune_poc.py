import copy
import functools
import uuid
from datetime import datetime
from typing import Optional

import hydra.utils
import pytorch_lightning as pl
import ray
import rul_datasets
import wandb
from ray import tune

import rul_adapt

FIXED_HPARAMS = ["_target_", "input_channels", "seq_len"]
BATCH_SIZE = 128


COMMON_SEARCH_SPACE = {
    "lr": tune.qloguniform(1e-5, 1e-2, 1e-5),  # quantized log uniform
    "fc_units": tune.choice([16, 32, 64, 128]),
    "feature_channels": tune.choice([16, 32, 64]),
}
CNN_SEARCH_SPACE = {
    "_target_": "rul_adapt.model.CnnExtractor",
    "num_layers": tune.randint(1, 10),
    "kernel_size": tune.choice([3, 5, 7]),
    "dropout": tune.quniform(0.0, 0.5, 0.1),  # quantized uniform
}
LSTM_SEARCH_SPACE = {
    "_target_": "rul_adapt.model.LstmExtractor",
    "num_layers": tune.randint(1, 3),
    "dropout": tune.quniform(0.0, 0.5, 0.1),  # quantized uniform
}


def tune_backbone(
    dataset: str,
    backbone: str,
    gpu: bool,
    entity: str,
    project: str,
    sweep_name: Optional[str],
):
    sweep_uuid = (
        str(uuid.uuid4()) if sweep_name is None else f"{sweep_name}-{datetime.now()}"
    )
    if backbone == "cnn":
        search_space = {**COMMON_SEARCH_SPACE, **CNN_SEARCH_SPACE}
    elif backbone == "lstm":
        search_space = {**COMMON_SEARCH_SPACE, **LSTM_SEARCH_SPACE}
    else:
        raise ValueError(f"Unknown backbone {backbone}.")

    if dataset == "cmapss":
        search_space["input_channels"] = 14  # fixes input channels
        if backbone == "cnn":
            search_space["seq_len"] = 30  # fixes sequence length for CNN
        source_config = {"_target_": "rul_datasets.CmapssReader", "window_size": 30}
        fds = list(range(1, 5))
        resources = {"gpu": 0.25}
    elif dataset == "femto":
        search_space["input_channels"] = 2  # fixes input channels
        if backbone == "cnn":
            search_space["seq_len"] = 2560  # fixes sequence length for CNN
        source_config = {"_target_": "rul_datasets.FemtoReader"}
        fds = list(range(1, 4))
        resources = {"gpu": 0.5}
    elif dataset == "xjtu-sy":
        search_space["input_channels"] = 2  # fixes input channels
        if backbone == "cnn":
            search_space["seq_len"] = 32768  # fixes sequence length for CNN
        source_config = {"_target_": "rul_datasets.XjtuSyReader"}
        fds = list(range(1, 4))
        resources = {"gpu": 0.5}
    else:
        raise ValueError(f"Unknown dataset {dataset}.")

    metric_columns = ["avg_rmse"] + [f"rmse_{i}" for i in fds]
    scheduler = tune.schedulers.FIFOScheduler()  # runs trials sequentially
    parameter_cols = [k for k in search_space.keys() if k not in FIXED_HPARAMS]
    reporter = tune.CLIReporter(  # prints progress to console
        parameter_columns=parameter_cols,
        metric_columns=metric_columns,
        max_column_length=15,
    )

    # set arguments constant for all trials and run tuning
    tune_func = functools.partial(
        run_training,
        source_config=source_config,
        fds=fds,
        sweep_uuid=sweep_uuid,
        entity=entity,
        project=project,
        gpu=gpu,
    )
    analysis = tune.run(
        tune_func,
        name=f"tune-{dataset}-{backbone}-supervised",
        metric="avg_rmse",  # monitor this metric
        mode="min",  # minimize the metric
        num_samples=100,
        resources_per_trial=resources if gpu else {"cpu": 4},
        scheduler=scheduler,
        config=search_space,
        progress_reporter=reporter,
        fail_fast=True,  # stop on first error
    )

    wandb.init(
        project=project,
        entity=entity,
        job_type="analysis",
        tags=[sweep_uuid],
    )
    analysis_table = wandb.Table(dataframe=analysis.dataframe())
    wandb.log({"tune_analysis": analysis_table})

    print("Best hyperparameters found were: ", analysis.best_config)


def run_training(config, source_config, fds, sweep_uuid, entity, project, gpu):
    config = copy.deepcopy(config)  # ray uses the config later and we modify it here

    if "seq_len" in config:
        if config["num_layers"] * (config["kernel_size"] - 1) >= config["seq_len"]:
            # reduce number of layers to fit sequence length
            config["num_layers"] = config["seq_len"] // (config["kernel_size"] - 1) - 1
    config["units"] = [config["feature_channels"]] * config["num_layers"]

    lr = config["lr"]

    # remove unnecessary hyperparameters for model instantiation (or hydra crashes)
    del config["num_layers"]
    del config["feature_channels"]
    del config["lr"]

    trial_uuid = uuid.uuid4()
    results = []
    for fd in fds:
        source_config["fd"] = fd
        source = hydra.utils.instantiate(source_config)
        dm = rul_datasets.RulDataModule(source, BATCH_SIZE)

        backbone = hydra.utils.instantiate(config)
        regressor = rul_adapt.model.FullyConnectedHead(
            config["fc_units"], [1], act_func_on_last_layer=False
        )
        approach = rul_adapt.approach.SupervisedApproach(
            loss_type="rmse", optim_type="adam", lr=lr
        )
        approach.set_model(backbone, regressor)

        logger = pl.loggers.WandbLogger(
            project=project,
            entity=entity,
            group=str(trial_uuid),
            tags=[sweep_uuid],
        )
        logger.experiment.define_metric("val/loss", summary="best", goal="minimize")
        callbacks = [
            pl.callbacks.EarlyStopping(monitor="val/loss", patience=20),
            pl.callbacks.ModelCheckpoint(monitor="val/loss", save_top_k=1),
        ]
        trainer = pl.Trainer(
            accelerator="gpu" if gpu else "cpu",
            max_epochs=100,
            logger=logger,
            callbacks=callbacks,
        )

        trainer.fit(approach, dm)
        results.append(trainer.checkpoint_callback.best_model_score.item())
        wandb.finish()

    # report average RMSE and RMSE for each FD
    tune.report(
        avg_rmse=sum(results) / len(results),
        **{f"rmse_{i}": r for i, r in enumerate(results, start=1)},
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cmapss")
    parser.add_argument("--backbone", type=str, default="cnn", choices=["cnn", "lstm"])
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--entity", type=str, default="adapt-rul")
    parser.add_argument("--project", type=str, default="test_supervised")
    parser.add_argument("--sweep_name", type=str, default=None)
    opt = parser.parse_args()

    ray.init(log_to_driver=False)
    tune_backbone(
        opt.dataset, opt.backbone, opt.gpu, opt.entity, opt.project, opt.sweep_name
    )
