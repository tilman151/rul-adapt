from typing import Any, Optional, Dict, Literal

import torch
import torchmetrics
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

import rul_adapt.loss
from rul_adapt.approach.abstract import AdaptionApproach


class DannApproach(AdaptionApproach):
    dann_loss: rul_adapt.loss.DomainAdversarialLoss

    def __init__(
        self,
        dann_factor: float,
        lr: float,
        weight_decay: float = 0.0,
        lr_decay_factor: Optional[float] = None,
        lr_decay_epochs: Optional[int] = None,
        loss_type: Literal["mae", "mse", "rmse"] = "mae",
    ):
        super().__init__()

        if not ((lr_decay_factor is None) == (lr_decay_epochs is None)):
            raise ValueError(
                "Either both lr_decay_factor and lr_decay_epoch "
                "need to be specified or neither of them."
            )

        self.dann_factor = dann_factor
        self.weight_decay = weight_decay
        self.lr = lr
        self.lr_decay_factor = lr_decay_factor
        self.lr_decay_epochs = lr_decay_epochs
        self.loss_type = loss_type

        # training metrics
        self.train_source_loss = self._get_train_source_loss()

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

    def _get_train_source_loss(self):
        if self.loss_type == "mae":
            train_source_loss = torchmetrics.MeanAbsoluteError()
        elif self.loss_type == "mse":
            train_source_loss = torchmetrics.MeanSquaredError()
        elif self.loss_type == "rmse":
            train_source_loss = torchmetrics.MeanSquaredError(squared=False)
        else:
            raise ValueError(
                f"Unknown loss type '{self.loss_type}'. "
                "Use either 'mae', 'mse' or 'rmse'."
            )

        return train_source_loss

    def set_model(
        self,
        feature_extractor: nn.Module,
        regressor: nn.Module,
        domain_disc: Optional[nn.Module] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if domain_disc is None:
            raise ValueError(
                "No domain discriminator was set. This approach is unlikely to work."
            )

        super().set_model(feature_extractor, regressor, *args, **kwargs)
        self.dann_loss = rul_adapt.loss.DomainAdversarialLoss(domain_disc)

    @property
    def domain_disc(self):
        """The domain discriminator network."""
        if hasattr(self, "dann_loss"):
            return self.dann_loss.domain_disc
        else:
            raise RuntimeError("Domain disc used before 'set_model' was called.")

    def configure_optimizers(self) -> Dict[str, Any]:
        optim = SGD(self.parameters(), self.lr, weight_decay=self.weight_decay)
        result: Dict[str, Any] = {"optimizer": optim}

        if (self.lr_decay_factor is not None) and (self.lr_decay_epochs is not None):
            scheduler = StepLR(optim, self.lr_decay_epochs, self.lr_decay_factor)
            result["lr_scheduler"] = {"scheduler": scheduler}

        return result

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.feature_extractor(inputs))

    def training_step(
        self, source: torch.Tensor, source_labels: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        source = self.feature_extractor(source)
        target = self.feature_extractor(target)

        rul_preds = self.regressor(source)
        mse_loss = self.train_source_loss(rul_preds, source_labels)

        dann_loss = self.dann_loss(source, target)

        loss = mse_loss + self.dann_factor * dann_loss

        self.log("train_loss", loss)
        self.log("train_source_loss", self.train_source_loss)
        self.log("train_dann", self.dann_loss)

        return loss

    def validation_step(
        self, features: torch.Tensor, labels: torch.Tensor, dataloader_idx: int
    ) -> None:
        predictions = self.forward(features)
        if dataloader_idx == 0:
            self.val_source_rmse(predictions, labels)
            self.val_source_score(predictions, labels)
            self.log("val_source_rmse", self.val_source_rmse)
            self.log("val_source_score", self.val_source_score)
        elif dataloader_idx == 1:
            self.val_target_rmse(predictions, labels)
            self.val_target_score(predictions, labels)
            self.log("val_target_rmse", self.val_target_rmse)
            self.log("val_target_score", self.val_target_score)
        else:
            raise RuntimeError(f"Unexpected val data loader idx {dataloader_idx}")

    def test_step(
        self, features: torch.Tensor, labels: torch.Tensor, dataloader_idx: int
    ) -> None:
        predictions = self.forward(features)
        if dataloader_idx == 0:
            self.test_source_rmse(predictions, labels)
            self.test_source_score(predictions, labels)
            self.log("test_source_rmse", self.test_source_rmse)
            self.log("test_source_score", self.test_source_score)
        elif dataloader_idx == 1:
            self.test_target_rmse(predictions, labels)
            self.test_target_score(predictions, labels)
            self.log("test_target_rmse", self.test_target_rmse)
            self.log("test_target_score", self.test_target_score)
        else:
            raise RuntimeError(f"Unexpected test data loader idx {dataloader_idx}")
