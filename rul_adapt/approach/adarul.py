import copy
from typing import Tuple, Optional, Any, List

import torch
import torchmetrics
from torch import nn

import rul_adapt
from rul_adapt.approach.abstract import AdaptionApproach
from rul_adapt.model import FullyConnectedHead


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

    CHECKPOINT_MODELS = ["_domain_disc", "frozen_feature_extractor"]

    _domain_disc: nn.Module
    frozen_feature_extractor: nn.Module

    def __init__(self, lr: float):
        super().__init__()

        self.lr = lr

        # training metrics
        self.gan_loss = nn.BCEWithLogitsLoss()

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
        layer and needs to use only a single output neuron.
        Args:
            feature_extractor: The feature extraction network.
            regressor: The RUL regression network.
            domain_disc: The domain discriminator network.
        """
        domain_disc = self._check_domain_disc(domain_disc)
        super().set_model(feature_extractor, regressor, *args, **kwargs)
        self._domain_disc = domain_disc
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
        if hasattr(self, "_domain_disc"):
            return self._domain_disc
        else:
            raise RuntimeError("Domain disc used before 'set_model' was called.")

    def configure_optimizers(self) -> List[torch.optim.Adam]:
        return [
            torch.optim.Adam(self.domain_disc.parameters(), self.lr),
            torch.optim.Adam(self.feature_extractor.parameters(), self.lr),
        ]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Predict the RUL values for a batch of input features."""
        return self.regressor(self.feature_extractor(inputs))

    def training_step(
        self, batch: List[torch.Tensor], batch_idx: int, optimizer_idx: int
    ) -> torch.Tensor:
        """
        Execute one training step.
        The `batch` argument is a list of three tensors representing the source
        features, source labels and target features. Both types of features are fed
        to the feature extractor. Then the regression loss for the source domain and
        the DANN loss between domains is computed. The regression, DANN and combined
        loss are logged.
        TODO: Update
        Args:
            batch: A list of a source feature, source label and target feature tensors.
            batch_idx: The index of the current batch.
            optimizer_idx: The index of the optimizer the loss is going to be used in.
        Returns:
            The combined loss.
        """
        source, _, target = batch

        if optimizer_idx == 0:
            loss = self._get_disc_loss(source, target)
            self.log("train/disc_loss", loss)
        elif optimizer_idx == 1:
            loss = self._get_gen_loss(target)
            self.log("train/gen_loss", loss)
        else:
            raise RuntimeError("Too many optimizers.")

        return loss

    def _get_disc_loss(self, source, target):
        batch_size = source.shape[0]
        source = self.frozen_feature_extractor(source).detach()
        target = self.feature_extractor(target).detach()
        domain_pred = self.domain_disc(torch.cat([source, target]))
        domain_labels = torch.cat(
            [
                torch.ones(batch_size, 1, device=self.device),  # real labels
                torch.zeros(batch_size, 1, device=self.device),  # fake labels
            ]
        )
        loss = self.gan_loss(domain_pred, domain_labels)

        return loss

    def _get_gen_loss(self, target):
        batch_size = target.shape[0]
        target = self.feature_extractor(target)
        domain_pred = self.domain_disc(target)
        domain_labels = torch.zeros(batch_size, 1, device=self.device)  # fake labels
        loss = -self.gan_loss(domain_pred, domain_labels)  # should maximize loss

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
