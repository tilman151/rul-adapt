import copy
import functools

import hydra.utils
import pytorch_lightning as pl
import ray
import rul_datasets
from ray import tune

import rul_adapt

LR = 1e-3
BATCH_SIZE = 128


COMMON_SEARCH_SPACE = {
    "fc_units": tune.choice([16, 32, 64, 128]),
    "feature_channels": tune.choice([16, 32, 64]),
}
CNN_SEARCH_SPACE = {
    "_target_": "rul_adapt.model.CnnExtractor",
    "num_layers": tune.randint(1, 10),
    "kernel_size": tune.choice([3, 5, 7]),
    "conv_dropout": tune.quniform(0.0, 0.5, 0.1),  # quantized uniform
}
LSTM_SEARCH_SPACE = {
    "_target_": "rul_adapt.model.LstmExtractor",
    "num_layers": tune.randint(1, 3),
    "lstm_dropout": tune.quniform(0.0, 0.5, 0.1),  # quantized uniform
}


def tune_backbone(dataset: str, backbone: str):
    if backbone == "cnn":
        search_space = {**COMMON_SEARCH_SPACE, **CNN_SEARCH_SPACE}
    elif backbone == "lstm":
        search_space = {**COMMON_SEARCH_SPACE, **LSTM_SEARCH_SPACE}
    else:
        raise ValueError(f"Unknown backbone {backbone}.")

    if dataset == "cmapss":
        search_space["input_channels"] = 14  # fixes input channels
        search_space["seq_len"] = 30  # fixes sequence length for CNN
        source_config = {"_target_": "rul_datasets.CmapssReader", "window_size": 30}
        fds = list(range(1, 5))
    elif dataset == "femto":
        search_space["input_channels"] = 2  # fixes input channels
        search_space["seq_len"] = 2560  # fixes sequence length for CNN
        source_config = {"_target_": "rul_datasets.FemtoReader"}
        fds = list(range(1, 4))
    elif dataset == "xjtu-sy":
        search_space["input_channels"] = 2  # fixes input channels
        search_space["seq_len"] = 32768  # fixes sequence length for CNN
        source_config = {"_target_": "rul_datasets.XjtuSyReader"}
        fds = list(range(1, 4))
    else:
        raise ValueError(f"Unknown dataset {dataset}.")

    metric_columns = ["avg_rmse"] + [f"rmse_{i}" for i in fds]
    scheduler = tune.schedulers.FIFOScheduler()  # runs trials sequentially
    reporter = tune.CLIReporter(  # prints progress to console
        parameter_columns=list(search_space.keys()), metric_columns=metric_columns
    )

    # set arguments constant for all trials and run tuning
    tune_func = functools.partial(
        run_training, source_config=source_config, fds=fds, backbone=backbone
    )
    analysis = tune.run(
        tune_func,
        name=f"tune-{dataset}-{backbone}-supervised",
        metric="avg_rmse",  # monitor this metric
        mode="min",  # minimize the metric
        num_samples=100,
        resources_per_trial={"gpu": 0.25},  # fits 4 trials on a single GPU
        scheduler=scheduler,
        config=search_space,
        progress_reporter=reporter,
        fail_fast=True,  # stop on first error
    )

    # TODO: where to log analysis to?
    print("Best hyperparameters found were: ", analysis.best_config)


def run_training(config, source_config, fds, backbone):
    config = copy.deepcopy(config)  # ray uses the config later and we modify it here

    # TODO: maybe unify argument names in models to avoid this if clause
    if backbone == "cnn":
        if config["num_layers"] * (config["kernel_size"] - 1) >= config["seq_len"]:
            # reduce number of layers to fit sequence length
            config["num_layers"] = config["seq_len"] // (config["kernel_size"] - 1) - 1
        config["conv_filters"] = [config["feature_channels"]] * config["num_layers"]
    elif backbone == "lstm":
        config["lstm_units"] = [config["feature_channels"]] * config["num_layers"]
        del config["seq_len"]

    # remove unnecessary hyperparameters for model instantiation (or hydra crashes)
    del config["num_layers"]
    del config["feature_channels"]

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
            loss_type="rmse", optim_type="adam", lr=LR
        )
        approach.set_model(backbone, regressor)

        logger = pl.loggers.WandbLogger(project="test_supervised", entity="adapt-rul")
        callbacks = [
            pl.callbacks.EarlyStopping(monitor="val/loss", patience=20),
            pl.callbacks.ModelCheckpoint(monitor="val/loss", save_top_k=1),
        ]
        trainer = pl.Trainer(
            accelerator="gpu", max_epochs=100, logger=logger, callbacks=callbacks
        )

        trainer.fit(approach, dm)
        results.append(trainer.checkpoint_callback.best_model_score.item())

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
    opt = parser.parse_args()

    ray.init(log_to_driver=False)
    tune_backbone(opt.dataset, opt.backbone)
