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
@pytest.mark.parametrize("feature_extractor", ["cnn", "lstm"])
class TestConfigComposition:
    @pytest.mark.parametrize("source_fd", ["one", "two", "three", "four"])
    @pytest.mark.parametrize("target_fd", ["one", "two", "three", "four"])
    def test_cmapss(
        self, source_fd, target_fd, adaption_mode, approach, feature_extractor
    ):
        self._check_no_missing_values(
            source_fd, target_fd, "cmapss", approach, feature_extractor, adaption_mode
        )

    @pytest.mark.parametrize("source_fd", ["one", "two", "three"])
    @pytest.mark.parametrize("target_fd", ["one", "two", "three"])
    @pytest.mark.parametrize("dataset", ["femto", "xjtu-sy"])
    def test_femto_xjtu_sy(
        self, source_fd, target_fd, dataset, adaption_mode, approach, feature_extractor
    ):
        self._check_no_missing_values(
            source_fd, target_fd, dataset, approach, feature_extractor, adaption_mode
        )

    def _check_no_missing_values(
        self, source_fd, target_fd, dataset, approach, feature_extractor, adaption_mode
    ):
        task_name = f"{source_fd}2{target_fd}"
        overrides = [
            f"+task={task_name}",
            f"+approach={approach}",
            f"+feature_extractor={feature_extractor}",
            f"+dataset={dataset}",
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
    def test_cmapss_run(
        self, source_fd, target_fd, adaption_mode, approach, feature_extractor, tmp_path
    ):
        self._check_runnable(
            source_fd,
            target_fd,
            "cmapss",
            approach,
            feature_extractor,
            adaption_mode,
            tmp_path,
        )

    @pytest.mark.integration
    @pytest.mark.parametrize("source_fd", ["one", "two", "three"])
    @pytest.mark.parametrize("target_fd", ["one", "two", "three"])
    @pytest.mark.parametrize("dataset", ["femto", "xjtu-sy"])
    def test_femto_xjtu_sy_run(
        self,
        source_fd,
        target_fd,
        dataset,
        adaption_mode,
        approach,
        feature_extractor,
        tmp_path,
    ):
        self._check_runnable(
            source_fd,
            target_fd,
            dataset,
            approach,
            feature_extractor,
            adaption_mode,
            tmp_path,
        )

    def _check_runnable(
        self,
        source_fd,
        target_fd,
        dataset,
        approach,
        feature_extractor,
        adaption_mode,
        tmp_path,
    ):
        if source_fd == target_fd:
            pytest.skip("source_fd == target_fd")
        with hydra.initialize(config_path="../config", version_base="1.2"):
            overrides = [
                f"+task={source_fd}2{target_fd}",
                f"+approach={approach}",
                f"+feature_extractor={feature_extractor}",
                f"+dataset={dataset}",
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
