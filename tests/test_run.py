import hydra.utils
import pytest
from omegaconf import OmegaConf

from rul_adapt import utils


@pytest.mark.parametrize("adaption_mode", ["transductive", "inductive"])
@pytest.mark.parametrize(
    "approach",
    [
        "adarul",
        "conditional_dann",
        "conditional_mmd",
        "consistency",
        "dann",
        "latent_align",
        "mmd",
        "pseudo_labels",
    ],
)
class TestConfigComposition:
    @pytest.mark.parametrize("source_fd", ["one", "two", "three", "four"])
    @pytest.mark.parametrize("target_fd", ["one", "two", "three", "four"])
    def test_cmapss(self, source_fd, target_fd, adaption_mode, approach):
        task_name = f"{source_fd}2{target_fd}"
        overrides = [
            f"+task={task_name}",
            "+feature_extractor=cnn",
            f"+approach={approach}",
            "+dataset=cmapss",
            f"adaption_mode={adaption_mode}",
        ]
        with hydra.initialize(config_path="../config", version_base="1.2"):
            if source_fd == target_fd:
                with pytest.raises(Exception):
                    _compose_and_resolve(overrides)
            else:
                _compose_and_resolve(overrides)

    @pytest.mark.integration
    @pytest.mark.parametrize("source_fd", ["one", "two", "three", "four"])
    @pytest.mark.parametrize("target_fd", ["one", "two", "three", "four"])
    def test_cmapss_run(self, source_fd, target_fd, adaption_mode, approach, tmp_path):
        if source_fd == target_fd:
            pytest.skip("source_fd == target_fd")
        with hydra.initialize(config_path="../config", version_base="1.2"):
            overrides = [
                f"+task={source_fd}2{target_fd}",
                "+feature_extractor=cnn",
                f"+approach={approach}",
                "+dataset=cmapss",
                f"adaption_mode={adaption_mode}",
                "pretraining.trainer.max_epochs=1",
                "training.trainer.max_epochs=1",
                "pretraining.trainer.logger=False",
                "training.trainer.logger=False",
                "accelerator=cpu",
                f"+pretraining.trainer.callbacks.0.dirpath={tmp_path.as_posix()}",
                f"+training.trainer.callbacks.0.dirpath={tmp_path.as_posix()}",
            ]
            config = _compose_and_resolve(overrides)
            runner = utils.str2callable(config["runner"], restriction="rul_adapt.run")
            runner(config)


def _compose_and_resolve(overrides):
    cfg = hydra.compose(config_name="config", overrides=overrides)
    cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    return cfg
