# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from dataclasses import dataclass

import logging
from pathlib import Path
from typing import Optional, Sequence, Dict, Any

import ray
from hydra.core.hydra_config import HydraConfig
from hydra.core.plugins import Plugins
from hydra.types import HydraContext
from hydra.core.config_store import ConfigStore
from hydra.core.singleton import Singleton
from hydra.core.utils import (
    JobReturn,
    configure_log,
    filter_overrides,
    run_job,
    setup_globals,
)
from hydra.plugins.launcher import Launcher
from hydra.types import TaskFunction
from omegaconf import DictConfig, open_dict

# IMPORTANT:
# If your plugin imports any module that takes more than a fraction of a second to import,
# Import the module lazily (typically inside launch()).
# Installed plugins are imported during Hydra initialization and plugins that are slow to import plugins will slow
# the startup of ALL hydra applications.
# Another approach is to place heavy includes in a file prefixed by _, such as _core.py:
# Hydra will not look for plugin in such files and will not import them during plugin discovery.


log = logging.getLogger(__name__)


@dataclass
class LauncherConfig:
    _target_: str = "hydra_plugins.ray_hydra_launcher.ray_launcher.SimpleRayLauncher"
    num_cpus: Optional[int] = None
    num_gpus: Optional[float] = None


ConfigStore.instance().store(group="hydra/launcher", name="ray", node=LauncherConfig)


class SimpleRayLauncher(Launcher):
    def __init__(self, num_cpus: Optional[int], num_gpus: Optional[int]) -> None:
        self.config: Optional[DictConfig] = None
        self.task_function: Optional[TaskFunction] = None
        self.hydra_context: Optional[HydraContext] = None

        self.num_cpus = num_cpus
        self.num_gpus = num_gpus

    def setup(
        self,
        *,
        hydra_context: HydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        self.config = config
        self.hydra_context = hydra_context
        self.task_function = task_function

    def launch(
        self, job_overrides: Sequence[Sequence[str]], initial_job_idx: int
    ) -> Sequence[JobReturn]:
        setup_globals()
        assert self.config is not None
        assert self.hydra_context is not None
        assert self.task_function is not None

        configure_log(self.config.hydra.hydra_logging, self.config.hydra.verbose)
        sweep_dir = Path(str(self.config.hydra.sweep.dir))
        sweep_dir.mkdir(parents=True, exist_ok=True)
        log.info(
            f"Example Launcher(num_cpus={self.num_cpus}, num_gpus={self.num_gpus}) "
            f"is launching {len(job_overrides)} jobs on ray"
        )
        log.info(f"Sweep output dir : {sweep_dir}")

        self.start_ray()

        runs = []
        for idx, overrides in enumerate(job_overrides):
            idx = initial_job_idx + idx
            lst = " ".join(filter_overrides(overrides))
            log.info(f"\t#{idx} : {lst}")
            sweep_config = self.hydra_context.config_loader.load_sweep_config(
                self.config, list(overrides)
            )
            with open_dict(sweep_config):
                sweep_config.hydra.job.id = f"job_id_for_{idx}"
                sweep_config.hydra.job.num = idx

                state = Singleton.get_state()
                remote_job = ray.remote(num_cpus=self.num_cpus, num_gpus=self.num_gpus)(
                    _run_job
                )
                ret = remote_job.remote(
                    hydra_context=self.hydra_context,
                    task_function=self.task_function,
                    sweep_config=sweep_config,
                    singleton_state=state,
                )
                runs.append(ret)

        results = []
        for i, run in enumerate(runs):
            results.append(ray.get(run))
            log.info(f"Finished #{i}")

        return results

    def start_ray(self) -> None:
        if not ray.is_initialized():
            log.info(f"Initializing ray")
            ray.init(log_to_driver=False, include_dashboard=True)
        else:
            log.info("Ray is already running.")


def _run_job(
    hydra_context: HydraContext,
    sweep_config: DictConfig,
    task_function: TaskFunction,
    singleton_state: Dict[Any, Any],
) -> JobReturn:
    setup_globals()
    Singleton.set_state(singleton_state)
    HydraConfig.instance().set_config(sweep_config)
    return run_job(
        hydra_context=hydra_context,
        task_function=task_function,
        config=sweep_config,
        job_dir_key="hydra.sweep.dir",
        job_subdir_key="hydra.sweep.subdir",
    )


def register_ray_launcher():
    Plugins.instance().register(SimpleRayLauncher)
