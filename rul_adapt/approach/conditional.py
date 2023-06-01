"""The Conditional Adaption approaches are derived from the [MMD] [
rul_adapt.approach.mmd] and [DANN][rul_adapt.approach.dann] approaches. They apply
their respective adaption loss not only to the whole data but also separately to
subsets of the data with a [ConditionalAdaptionLoss]
[rul_adapt.loss.conditional.ConditionalAdaptionLoss]. Fuzzy sets with rectangular
membership functions define these subsets.

Both variants were introduced by
[Cheng et al.](https://doi.org/10.1007/s10845-021-01814-y) in 2021."""

from copy import deepcopy
from typing import List, Tuple, Literal, Optional, Any

import torch
from torch import nn

import rul_adapt
from rul_adapt import utils
from rul_adapt.approach.abstract import AdaptionApproach
from rul_adapt.approach.evaluation import AdaptionEvaluator
from rul_adapt.model import FullyConnectedHead


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

        self.evaluator = AdaptionEvaluator(self.forward, self.log)

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
        the MMD loss and the conditional MMD loss are computed. The
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
        domain = utils.dataloader2domain(dataloader_idx)
        self.evaluator.validation(batch, domain)

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
        domain = utils.dataloader2domain(dataloader_idx)
        self.evaluator.test(batch, domain)


class ConditionalDannApproach(AdaptionApproach):
    CHECKPOINT_MODELS = ["dann_loss", "conditional_dann_loss"]

    dann_loss: rul_adapt.loss.DomainAdversarialLoss
    conditional_dann_loss: rul_adapt.loss.ConditionalAdaptionLoss

    def __init__(
        self,
        lr: float,
        dann_factor: float,
        dynamic_adaptive_factor: float,
        fuzzy_sets: List[Tuple[float, float]],
        loss_type: Literal["mse", "rmse", "mae"] = "mae",
    ) -> None:
        super().__init__()

        self.lr = lr
        self.dann_factor = dann_factor
        self.dynamic_adaptive_factor = dynamic_adaptive_factor
        self.loss_type = loss_type
        self._fuzzy_sets = fuzzy_sets

        self.train_source_loss = utils.get_loss(self.loss_type)
        self.evaluator = AdaptionEvaluator(self.forward, self.log)

        self.save_hyperparameters()

    @property
    def fuzzy_sets(self) -> List[Tuple[float, float]]:
        return self._fuzzy_sets

    def set_model(
        self,
        feature_extractor: nn.Module,
        regressor: nn.Module,
        domain_disc: Optional[nn.Module] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Set the feature extractor, regressor, and domain discriminator for this
        approach.

        The discriminator is not allowed to have an activation function on its last
        layer and needs to use only a single output neuron.
        It is wrapped by a
        [DomainAdversarialLoss][rul_adapt.loss.DomainAdversarialLoss].

        A copy of the discriminator is used for each conditional loss governing a
        fuzzy set.

        Args:
            feature_extractor: The feature extraction network.
            regressor: The RUL regression network.
            domain_disc: The domain discriminator network.
                         Copied for each fuzzy set.
        """
        domain_disc = self._check_domain_disc(domain_disc)
        super().set_model(feature_extractor, regressor, *args, **kwargs)
        self.dann_loss = rul_adapt.loss.DomainAdversarialLoss(domain_disc)
        cond_losses = [deepcopy(self.dann_loss) for _ in range(len(self.fuzzy_sets))]
        self.conditional_dann_loss = rul_adapt.loss.ConditionalAdaptionLoss(
            cond_losses, self.fuzzy_sets
        )

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
        the MMD loss and the conditional MMD loss are computed. The
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
        dann_loss = self.dann_loss(source, target)
        cond_dann_loss = self.conditional_dann_loss(
            source, source_preds, target, target_preds
        )
        combined_dann_loss = (1 - daf) * dann_loss + daf * cond_dann_loss
        loss = source_loss + self.dann_factor * combined_dann_loss

        self.log("train/loss", loss)
        self.log("train/source_loss", self.train_source_loss)
        self.log("train/dann", self.dann_loss)
        self.log("train/conditional_dann", self.conditional_dann_loss)

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
        domain = utils.dataloader2domain(dataloader_idx)
        self.evaluator.validation(batch, domain)

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
        domain = utils.dataloader2domain(dataloader_idx)
        self.evaluator.test(batch, domain)
