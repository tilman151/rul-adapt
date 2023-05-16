from typing import List, Tuple, Literal

import torch
import torchmetrics

import rul_adapt
from rul_adapt import utils
from rul_adapt.approach.abstract import AdaptionApproach


class ConditionalMmdApproach(AdaptionApproach):
    def __init__(
        self,
        lr: float,
        mmd_factor: float,
        num_mmd_kernels: int,
        dynamic_adaptive_factor: float,
        fuzzy_sets: List[Tuple[float, float]],
        loss_type: Literal["mse", "rmse", "mae"] = "mae",
    ) -> None:
        super().__init__()

        self.lr = lr
        self.mmd_factor = mmd_factor
        self.num_mmd_kernels = num_mmd_kernels
        self.dynamic_adaptive_factor = dynamic_adaptive_factor
        self.loss_type = loss_type

        # training metrics
        self.train_source_loss = utils.get_loss(self.loss_type)
        self.mmd_loss = rul_adapt.loss.MaximumMeanDiscrepancyLoss(self.num_mmd_kernels)
        conditional_mmd_losses = [
            rul_adapt.loss.MaximumMeanDiscrepancyLoss(self.num_mmd_kernels)
            for _ in range(len(fuzzy_sets))
        ]
        self.conditional_mmd_loss = rul_adapt.loss.ConditionalAdaptionLoss(
            conditional_mmd_losses, fuzzy_sets, mean_over_sets=True
        )

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

    @property
    def fuzzy_sets(self) -> List[Tuple[float, float]]:
        return self.conditional_mmd_loss.fuzzy_sets

    def configure_optimizers(self) -> torch.optim.Adam:
        """Configure an Adam optimizer."""
        return torch.optim.Adam(self.parameters(), self.lr)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Predict the RUL values for a batch of input features."""
        return self.regressor(self.feature_extractor(inputs))

    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Execute one training step.

        The `batch` argument is a list of three tensors representing the source
        features, source labels and target features. Both types of features are fed
        to the feature extractor. Then the regression loss for the source domain,
        the MMD loss and the conditional MMD loss is computed. The
        regression, MMD, conditional MMD and combined loss are logged.

        Args:
            batch: A list of a source feature, source label and target feature tensors.
            batch_idx: The index of the current batch.
        Returns:
            The combined loss.
        """
        source, source_labels, target = batch
        source_labels = source_labels[:, None]
        daf = self.dynamic_adaptive_factor

        source = self.feature_extractor(source)
        target = self.feature_extractor(target)
        source_preds = self.regressor(source)
        target_preds = self.regressor(target)

        source_loss = self.train_source_loss(source_preds, source_labels)
        mmd_loss = self.mmd_loss(source, target)
        cond_mmd_loss = self.conditional_mmd_loss(
            source, source_preds, target, target_preds
        )
        combined_mmd_loss = (1 - daf) * mmd_loss + daf * cond_mmd_loss
        loss = source_loss + self.mmd_factor * combined_mmd_loss

        self.log("train/loss", loss)
        self.log("train/source_loss", self.train_source_loss)
        self.log("train/mmd", self.mmd_loss)
        self.log("train/conditional_mmd", self.conditional_mmd_loss)

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


class ConditionalDomainDannApproach(AdaptionApproach):
    pass
