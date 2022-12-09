from itertools import chain

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
        max_rul: int,
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
        self.level_align = rul_adapt.loss.DegradationLevelRegularizationLoss(max_rul)
        self.fusion_align = rul_adapt.loss.MaximumMeanDiscrepancyLoss(num_kernels=5)

        # validation metrics
        self.val_source_mse = torchmetrics.MeanSquaredError()
        self.val_target_mse = torchmetrics.MeanSquaredError()

        # testing metrics
        self.test_source_mse = torchmetrics.MeanSquaredError()
        self.test_target_mse = torchmetrics.MeanSquaredError()

        self.save_hyperparameters()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        params = chain(self.feature_extractor.parameters(), self.regressor.parameters())
        optim = torch.optim.Adam(params, self.lr)

        return optim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.feature_extractor(features))

    def training_step(
        self,
        healthy: torch.Tensor,
        source: torch.Tensor,
        source_labels: torch.Tensor,
        target: torch.Tensor,
        target_degradation_steps: torch.Tensor,
    ) -> torch.Tensor:
        healthy = self.feature_extractor(healthy)
        source = self.feature_extractor(source)
        target = self.feature_extractor(target)

        rul_predictions = self.regressor(source)
        mse_loss = self.train_mse(rul_predictions, source_labels)

        healthy_loss = self.healthy_align(healthy)
        direction_loss = self.direction_align(healthy, torch.cat([source, target]))
        level_loss = self.level_align(
            healthy, source, source_labels, target, target_degradation_steps
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
        self.log("fusion_align", fusion_loss)

        return loss

    def validation_step(
        self, features: torch.Tensor, labels: torch.Tensor, dataloader_idx: int
    ) -> None:
        predictions = self.forward(features)
        if dataloader_idx == 0:
            self.val_source_mse(predictions, labels)
            self.log("val_source_mse", self.val_source_mse)
        elif dataloader_idx == 1:
            self.val_target_mse(predictions, labels)
            self.log("val_target_mse", self.val_target_mse)
        else:
            raise RuntimeError(f"Unexpected val data loader idx {dataloader_idx}")

    def test_step(
        self, features: torch.Tensor, labels: torch.Tensor, dataloader_idx: int
    ) -> None:
        predictions = self.forward(features)
        if dataloader_idx == 0:
            self.test_source_mse(predictions, labels)
            self.log("test_source_mse", self.test_source_mse)
        elif dataloader_idx == 1:
            self.test_target_mse(predictions, labels)
            self.log("test_target_mse", self.test_target_mse)
        else:
            raise RuntimeError(f"Unexpected test data loader idx {dataloader_idx}")
