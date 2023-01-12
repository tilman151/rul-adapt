import copy
import math
from itertools import chain
from typing import Optional, Any, List

import torch
import torchmetrics
from torch import nn

import rul_adapt.loss
from rul_adapt.approach.abstract import AdaptionApproach
from rul_adapt.model import FullyConnectedHead


class ConsistencyApproachPretraining(AdaptionApproach):
    def __init__(self, lr: float):
        super().__init__()

        self.lr = lr

        self.train_loss = torchmetrics.MeanSquaredError(squared=False)
        self.val_loss = torchmetrics.MeanSquaredError(squared=False)

        self.save_hyperparameters()

    def configure_optimizers(self) -> torch.optim.SGD:
        return torch.optim.SGD(self.parameters(), lr=self.lr)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.feature_extractor(inputs))

    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
        inputs, labels = batch
        predictions = self.forward(inputs)
        loss = self.train_loss(predictions, labels[:, None])
        self.log("train/loss", self.train_loss)

        return loss

    def validation_step(self, batch: List[torch.Tensor], batch_idx: int) -> None:
        inputs, labels = batch
        predictions = self.forward(inputs)
        self.val_loss(predictions, labels[:, None])
        self.log("val/loss", self.val_loss)


class ConsistencyApproach(AdaptionApproach):
    dann_loss: rul_adapt.loss.DomainAdversarialLoss
    frozen_feature_extractor: nn.Module

    def __init__(self, consistency_factor: float, lr: float, max_epochs: int):
        super().__init__()

        self.consistency_factor = consistency_factor
        self.lr = lr
        self.max_epochs = max_epochs

        # training metrics
        self.train_source_loss = torchmetrics.MeanSquaredError(squared=False)
        self.consistency_loss = rul_adapt.loss.ConsistencyLoss()

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

    def set_model(
        self,
        feature_extractor: nn.Module,
        regressor: nn.Module,
        domain_disc: Optional[nn.Module] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Set the feature extractor, regressor and domain discriminator for this approach.
        The discriminator is not allowed to have an activation function on its last
        layer and needs to use only a single output neuron. It is wrapped by a
        [DomainAdversarialLoss][rul_adapt.loss.DomainAdversarialLoss].
        TODO: Add explanation of frozen FE
        Args:
            feature_extractor: The feature extraction network.
            regressor: The RUL regression network.
            domain_disc: The domain discriminator network.
        """
        domain_disc = self._check_domain_disc(domain_disc)
        super().set_model(feature_extractor, regressor, *args, **kwargs)
        self.dann_loss = rul_adapt.loss.DomainAdversarialLoss(domain_disc)
        self.frozen_feature_extractor = copy.deepcopy(feature_extractor)
        self.frozen_feature_extractor.requires_grad_(False)  # freeze network

    def _check_domain_disc(self, domain_disc: Optional[nn.Module]) -> nn.Module:
        if domain_disc is None:
            raise ValueError(
                "No domain discriminator was set. This approach is unlikely to work."
            )
        if (
            isinstance(domain_disc, FullyConnectedHead)
            and domain_disc.act_func_on_last_layer
        ):
            raise ValueError(
                "Domain discriminator has an activation function on its last layer. "
                "This is not allowed due to torch.nn.BCEWithLogitsLoss being used as "
                "its loss. Please set 'act_func_on_last_layer' to False."
            )

        return domain_disc

    @property
    def domain_disc(self):
        """The domain discriminator network."""
        if hasattr(self, "dann_loss"):
            return self.dann_loss.domain_disc
        else:
            raise RuntimeError("Domain disc used before 'set_model' was called.")

    @property
    def dann_factor(self):
        return 2 / (1 + math.exp(-10 * self.current_epoch / self.max_epochs)) - 1

    def configure_optimizers(self) -> torch.optim.SGD:
        parameters = chain(
            self.feature_extractor.parameters(),
            self.regressor.parameters(),
            self.dann_loss.parameters(),
        )  # excludes frozen_feature_extractor from optimization
        optim = torch.optim.SGD(parameters, lr=self.lr)

        return optim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.feature_extractor(inputs))

    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
        source, source_labels, target = batch
        source_labels = source_labels[:, None]

        frozen_source = self.frozen_feature_extractor(source)
        source = self.feature_extractor(source)
        target = self.feature_extractor(target)

        rul_preds = self.regressor(source)
        rmse_loss = self.train_source_loss(rul_preds, source_labels)

        dann_loss = self.dann_loss(source, target)
        consistency_loss = self.consistency_loss(frozen_source, source)

        loss = (
            rmse_loss
            + self.dann_factor * dann_loss
            + self.consistency_factor * consistency_loss
        )

        self.log("train/loss", loss)
        self.log("train/source_loss", self.train_source_loss)
        self.log("train/dann", self.dann_loss)
        self.log("train/consistency", self.consistency_loss)

        return loss

    def on_train_epoch_start(self) -> None:
        self.log("train/dann_factor", self.dann_factor)

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
