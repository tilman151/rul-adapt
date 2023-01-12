from typing import Tuple

import torch
import torchmetrics

from rul_adapt.approach.abstract import AdaptionApproach


class AdaRulApproachPretraining(AdaptionApproach):
    def __init__(self, lr: float):
        super().__init__()

        self.lr = lr

        self.train_loss = torchmetrics.MeanSquaredError()
        self.val_loss = torchmetrics.MeanSquaredError(squared=False)

    def configure_optimizers(self) -> torch.optim.Adam:
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.feature_extractor(inputs))

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        inputs, labels = batch
        predictions = self.forward(inputs)
        loss = self.train_loss(predictions, labels[:, None])
        self.log("train/loss", self.train_loss)

        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        inputs, labels = batch
        predictions = self.forward(inputs)
        self.val_loss(predictions, labels[:, None])
        self.log("val/loss", self.val_loss)


class AdaRulApproach(AdaptionApproach):
    pass
