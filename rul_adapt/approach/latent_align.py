from typing import Tuple, List

import torch
import torchmetrics

import rul_adapt
from rul_adapt.approach.abstract import AdaptionApproach


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

        self.log("loss", loss)
        self.log("mse", self.train_mse)
        self.log("healthy_align", self.healthy_align)
        self.log("direction_align", self.direction_align)
        self.log("level_align", self.level_align)
        self.log("fusion_align", self.fusion_align)

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
