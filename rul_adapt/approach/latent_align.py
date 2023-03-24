from typing import Tuple, List, Any, Optional

import numpy as np
import torch
import torchmetrics
from torch import nn

import rul_adapt
from rul_datasets import utils
from rul_adapt.approach.abstract import AdaptionApproach


class LatentAlignFttpApproach(AdaptionApproach):
    CHECKPOINT_MODELS = ["_generator"]

    _generator: nn.Module

    def __init__(self, lr: float, noise_dim: int):
        super().__init__()

        self.lr = lr
        self.noise_dim = noise_dim

        self.gan_loss = torch.nn.BCEWithLogitsLoss()
        self.grl = rul_adapt.loss.adaption.GradientReversalLayer()

        self.save_hyperparameters()

    def set_model(
        self,
        feature_extractor: nn.Module,
        regressor: nn.Module,
        generator: Optional[nn.Module] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().set_model(feature_extractor, regressor)
        if generator is None:
            raise ValueError("Generator not set. This approach is unlikely to work.")
        self._generator = generator

    @property
    def generator(self):
        """The generator network."""
        if hasattr(self, "_generator"):
            return self._generator
        else:
            raise RuntimeError("Generator used before 'set_model' was called.")

    def configure_optimizers(self) -> torch.optim.Adam:
        return torch.optim.Adam(self.parameters(), self.lr)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.feature_extractor(inputs))

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        features, _ = batch
        batch_size = features.shape[0]

        pred_real = self.forward(features)
        loss_real = self.gan_loss(
            pred_real, torch.zeros(batch_size, 1, device=self.device)
        )

        noise = torch.randn(batch_size, 1, self.noise_dim, device=self.device)
        fake_features = self.grl(self.generator(noise)).reshape_as(features)
        pred_fake = self.forward(fake_features)
        loss_fake = self.gan_loss(
            pred_fake, torch.ones(batch_size, 1, device=self.device)
        )

        loss = (loss_real + loss_fake) / 2
        self.log("train/loss", loss)

        return loss


@torch.no_grad()
def get_first_time_to_predict(
    fttp_model: LatentAlignFttpApproach,
    features: np.ndarray,
    window_size: int,
    chunk_size: int,
    healthy_index: int,
    threshold_coefficient: float,
) -> int:
    if threshold_coefficient <= 1:
        raise ValueError("Threshold coefficient needs to be greater than one.")

    health_indicator = get_health_indicator(
        fttp_model, features, window_size, chunk_size
    )
    offset = len(features) - len(health_indicator)  # windowing cuts off first windows
    healthy = np.mean(health_indicator[: healthy_index - offset])
    over_thresh = np.argwhere(health_indicator > threshold_coefficient * healthy)

    if len(over_thresh) == 0:
        raise RuntimeError("Health indicator never passes threshold.")
    fttp = over_thresh[0, 0] + offset

    return fttp


def get_health_indicator(
    fttp_model: nn.Module, features: np.ndarray, window_size: int, chunk_size: int
) -> np.ndarray:
    chunked = extract_chunk_windows(features, window_size, chunk_size)
    health_indicator = []
    chunks_per_window = features.shape[1] // chunk_size
    for batch in np.split(chunked, len(chunked) // chunks_per_window):
        preds = fttp_model(utils.feature_to_tensor(batch, torch.float))
        health_indicator.append(np.std(preds.detach().numpy()))
    health_indicator = np.array(health_indicator)

    return health_indicator


def extract_chunk_windows(
    features: np.ndarray, window_size: int, chunk_size: int
) -> np.ndarray:
    old_window_size = features.shape[1]
    window_multiplier = old_window_size // chunk_size
    num_new_windows = (features.shape[0] - window_size + 1) * window_multiplier

    chunk_idx = np.tile(np.arange(chunk_size), num_new_windows * window_size)
    intra_offsets = np.tile(np.arange(window_size), num_new_windows) * old_window_size
    inter_offsets = np.repeat(np.arange(num_new_windows), window_size) * chunk_size
    offsets = np.repeat(intra_offsets + inter_offsets, chunk_size)
    window_idx = chunk_idx + offsets

    flat_features = features.reshape((-1, features.shape[2]))
    flat_windows = flat_features[window_idx]
    windows = flat_windows.reshape((num_new_windows, window_size * chunk_size, -1))

    return windows


class LatentAlignApproach(AdaptionApproach):
    def __init__(
        self,
        alpha_healthy: float,
        alpha_direction: float,
        alpha_level: float,
        alpha_fusion: float,
        lr: float,
    ):
        super().__init__()

        self.alpha_healthy = alpha_healthy
        self.alpha_direction = alpha_direction
        self.alpha_level = alpha_level
        self.alpha_fusion = alpha_fusion
        self.lr = lr

        # training metrics
        self.train_mse = torchmetrics.MeanSquaredError()
        self.healthy_align = rul_adapt.loss.HealthyStateAlignmentLoss()
        self.direction_align = rul_adapt.loss.DegradationDirectionAlignmentLoss()
        self.level_align = rul_adapt.loss.DegradationLevelRegularizationLoss()
        self.fusion_align = rul_adapt.loss.MaximumMeanDiscrepancyLoss(num_kernels=5)

        # validation metrics
        self.val_source_rmse = torchmetrics.MeanSquaredError(squared=False)
        self.val_target_rmse = torchmetrics.MeanSquaredError(squared=False)
        self.val_source_score = rul_adapt.loss.RULScore()
        self.val_target_score = rul_adapt.loss.RULScore()

        # testing metrics
        self.test_source_rmse = torchmetrics.MeanSquaredError(squared=False)
        self.test_target_rmse = torchmetrics.MeanSquaredError(squared=False)
        self.test_source_score = rul_adapt.loss.RULScore()
        self.test_target_score = rul_adapt.loss.RULScore()

        self.save_hyperparameters()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optim = torch.optim.Adam(self.parameters(), self.lr)

        return optim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.feature_extractor(features))

    def training_step(
        self, batch: Tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor:
        source, source_degradation_steps, source_labels, *_ = batch
        *_, target, target_degradation_steps, healthy = batch
        source_labels = source_labels[:, None]

        healthy = self.feature_extractor(healthy)
        source = self.feature_extractor(source)
        target = self.feature_extractor(target)

        rul_predictions = self.regressor(source)
        mse_loss = self.train_mse(rul_predictions, source_labels)

        healthy_loss = self.healthy_align(healthy)
        direction_loss = self.direction_align(healthy, torch.cat([source, target]))
        level_loss = self.level_align(
            healthy, source, source_degradation_steps, target, target_degradation_steps
        )
        fusion_loss = self.fusion_align(source, target)

        loss = (
            mse_loss
            + self.alpha_healthy * healthy_loss
            + self.alpha_direction * direction_loss
            + self.alpha_level * level_loss
            + self.alpha_fusion * fusion_loss
        )

        self.log("train/loss", loss)
        self.log("train/mse", self.train_mse)
        self.log("train/healthy_align", self.healthy_align)
        self.log("train/direction_align", self.direction_align)
        self.log("train/level_align", self.level_align)
        self.log("train/fusion_align", self.fusion_align)

        return loss

    def validation_step(
        self, batch: List[torch.Tensor], batch_idx: int, dataloader_idx: int
    ) -> None:
        """
        Execute one validation step.

        The `batch` argument is a list of two tensors representing features and
        labels. A RUL prediction is made from the features and the validation RMSE
        and RUL score are calculated. The metrics recorded for dataloader idx zero
        are assumed to be from the source domain and for dataloader idx one from the
        target domain. The metrics are written to the configured logger under the
        prefix `val`.

        Args:
            batch: A list containing a feature and a label tensor.
            batch_idx: The index of the current batch.
            dataloader_idx: The index of the current dataloader (0: source, 1: target).
        """
        features, labels = batch
        labels = labels[:, None]
        predictions = self.forward(features)
        if dataloader_idx == 0:
            self.val_source_rmse(predictions, labels)
            self.val_source_score(predictions, labels)
            self.log("val/source_rmse", self.val_source_rmse)
            self.log("val/source_score", self.val_source_score)
        elif dataloader_idx == 1:
            self.val_target_rmse(predictions, labels)
            self.val_target_score(predictions, labels)
            self.log("val/target_rmse", self.val_target_rmse)
            self.log("val/target_score", self.val_target_score)
        else:
            raise RuntimeError(f"Unexpected val data loader idx {dataloader_idx}")

    def test_step(
        self, batch: List[torch.Tensor], batch_idx: int, dataloader_idx: int
    ) -> None:
        """
        Execute one test step.

        The `batch` argument is a list of two tensors representing features and
        labels. A RUL prediction is made from the features and the validation RMSE
        and RUL score are calculated. The metrics recorded for dataloader idx zero
        are assumed to be from the source domain and for dataloader idx one from the
        target domain. The metrics are written to the configured logger under the
        prefix `test`.

        Args:
            batch: A list containing a feature and a label tensor.
            batch_idx: The index of the current batch.
            dataloader_idx: The index of the current dataloader (0: source, 1: target).
        """
        features, labels = batch
        labels = labels[:, None]
        predictions = self.forward(features)
        if dataloader_idx == 0:
            self.test_source_rmse(predictions, labels)
            self.test_source_score(predictions, labels)
            self.log("test/source_rmse", self.test_source_rmse)
            self.log("test/source_score", self.test_source_score)
        elif dataloader_idx == 1:
            self.test_target_rmse(predictions, labels)
            self.test_target_score(predictions, labels)
            self.log("test/target_rmse", self.test_target_rmse)
            self.log("test/target_score", self.test_target_score)
        else:
            raise RuntimeError(f"Unexpected test data loader idx {dataloader_idx}")
