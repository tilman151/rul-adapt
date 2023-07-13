import uuid
from datetime import datetime
from functools import partial
from itertools import product
from typing import Optional

import hydra
import ray
import wandb
from omegaconf import OmegaConf

import rul_adapt.tune
from rul_adapt import utils


INT2STR = {1: "one", 2: "two", 3: "three", 4: "four"}


def tune_adaption(
    dataset: str,
    backbone: str,
    approach: str,
    gpu: str,
    entity: str,
    project: str,
    sweep_name: Optional[str],
):
    sweep_uuid = (
        str(uuid.uuid4()) if sweep_name is None else f"{sweep_name}-{datetime.now()}"
    )
    search_space = getattr(rul_adapt.tune, approach)
    base_overrides = [
        f"+feature_extractor={backbone}",
        f"+approach={approach}",
        f"+dataset={dataset}",
        f"logger.entity={entity}",
        f"logger.project={project}",
        f"+logger.tags=[{sweep_uuid}]",
        f"accelerator={'gpu' if gpu else 'cpu'}",
    ]
    if dataset == "cmapss":
        fds = [1, 2, 3, 4]
        resources = {"gpu": 0.25}
    elif dataset == "femto" or dataset == "xjtu-sy":
        fds = [1, 2, 3]
        resources = {"gpu": 1.0}
    else:
        raise ValueError(f"Unknown dataset {dataset}.")

    metric_columns = ["avg_rmse"] + [f"rmse_{i}" for i in fds]
    scheduler = ray.tune.schedulers.FIFOScheduler()  # runs trials sequentially
    reporter = ray.tune.CLIReporter(  # prints progress to console
        parameter_columns=search_space.keys(),
        metric_columns=metric_columns,
        max_column_length=15,
    )
    tune_func = partial(run_training, base_overrides=base_overrides, fds=fds)
    analysis = ray.tune.run(
        tune_func,
        name=f"tune-{dataset}-{backbone}-{approach}",
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


def run_training(config, base_overrides, fds):
    logger_overrides = [f"+logger.group={uuid.uuid4()}"]
    hparam_overrides = rul_adapt.tune.config2overrides(config)
    base_overrides = base_overrides + hparam_overrides + logger_overrides
    results = {}
    for source_fd, target_fd in product(fds, fds):
        if source_fd == target_fd:
            continue
        with hydra.initialize(config_path="config", version_base="1.2"):
            task_override = [f"+task={INT2STR[source_fd]}2{INT2STR[target_fd]}"]
            overrides = task_override + base_overrides
            run_config = hydra.compose(config_name="config", overrides=overrides)
            run_config = OmegaConf.to_container(
                run_config, resolve=True, throw_on_missing=True
            )
            runner = utils.str2callable(
                run_config["runner"], restriction="rul_adapt.run"
            )
            run_result = runner(run_config)
            results[f"rmse_{source_fd}_{target_fd}"] = run_result

    # report average RMSE and RMSE for each FD
    ray.tune.report(
        avg_rmse=sum(results.values()) / len(results),
        **results,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cmapss")
    parser.add_argument("--backbone", type=str, default="cnn", choices=["cnn", "lstm"])
    parser.add_argument("--approach", type=str, default="dann", choices=["dann"])
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--entity", type=str, default="adapt-rul")
    parser.add_argument("--project", type=str, default="approach-tuning")
    parser.add_argument("--sweep_name", type=str, default=None)
    opt = parser.parse_args()

    ray.init(log_to_driver=False)
    tune_adaption(
        opt.dataset,
        opt.backbone,
        opt.approach,
        opt.gpu,
        opt.entity,
        opt.project,
        opt.sweep_name,
    )
