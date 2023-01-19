"""The Consistency DANN approach uses a consistency loss in tandem with a [DANN
loss][rul_adapt.approach.dann]. First, the network is pre-trained in a supervised
fashion on the source domain. The pre-trained weights are then used to initialize the
main training stage. The consistency loss penalizes the weights of the feature
extractor moving away from the pre-trained version. This way the feature extractor
weights stay close to the pre-trained weights.

```python
# pre-training stage

Source --> PreFeatEx --> Source Feats --> Regressor  --> RUL Prediction

# main training stage

   ------- PreTrainFeatEx --> PreTrain Source Feats --> Consistency Loss
   |
   |
Source --> FeatEx --> Source Feats -----------> Regressor  --> RUL Prediction
        ^         |                 |
        |         |                 v
Target --         --> Target Feats -->  GRL --> DomainDisc --> Domain Prediction
```

This version of DANN was introduced by
[Siahpour et al.](https://doi.org/10.1109/TIM.2022.3162283)."""

import copy
import math
from itertools import chain
from typing import Optional, Any, List

import numpy as np
import torch
import torchmetrics
from torch import nn

import rul_adapt.loss
from rul_adapt.approach.abstract import AdaptionApproach
from rul_adapt.model import FullyConnectedHead


class ConsistencyApproachPretraining(AdaptionApproach):
    """The pre-training for the Consistency DANN approach uses an RMSE loss to train
    a feature extractor and regressor in a supervised fashion on the source domain.
    After pre-training the network weights are used to initialize the main training
    stage.

    The regressor needs the same number of input units as the feature extractor has
    output units.

    Examples:
        ```pycon
        >>> from rul_adapt import model
        >>> from rul_adapt import approach
        >>> feat_ex = model.CnnExtractor(1, [16, 16, 1], 10, fc_units=16)
        >>> reg = model.FullyConnectedHead(16, [1])
        >>> disc = model.FullyConnectedHead(16, [8, 1], act_func_on_last_layer=False)
        >>> pre = approach.ConsistencyApproachPretraining(0.01)
        >>> pre.set_model(feat_ex, reg, disc)
        >>> main = approach.ConsistencyApproach(1.0, 0.01)
        >>> main.set_model(pre.feature_extractor, pre.regressor, disc)
        ```
    """

    def __init__(self, lr: float):
        """
        Create a new Consistency DANN pretraining approach.

        Args:
            lr: Learning rate.
        """
        super().__init__()

        self.lr = lr

        self.train_loss = torchmetrics.MeanSquaredError(squared=False)
        self.val_loss = torchmetrics.MeanSquaredError(squared=False)

        self.save_hyperparameters()

    def configure_optimizers(self) -> torch.optim.SGD:
        """Configure an SGD optimizer with the specified learning rate."""
        return torch.optim.SGD(self.parameters(), lr=self.lr)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Predict the RUL values for a batch of input features."""
        return self.regressor(self.feature_extractor(inputs))

    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Execute one training step.

        The `batch` argument is a list of two tensors representing features and
        labels. The features are used to predict RUL values that are compared against
        the labels with an RMSE loss. The loss is then logged.

        Args:
            batch: A list of feature and label tensors.
            batch_idx: The index of the current batch.
        Returns:
            The RMSE loss.
        """
        inputs, labels = batch
        predictions = self.forward(inputs)
        loss = self.train_loss(predictions, labels[:, None])
        self.log("train/loss", self.train_loss)

        return loss

    def validation_step(self, batch: List[torch.Tensor], batch_idx: int) -> None:
        """
        Execute one validation step.

        The `batch` argument is a list of two tensors representing features and
        labels. The features are used to predict RUL values that are compared against
        the labels with an RMSE loss. The loss is then logged.

        Args:
            batch: A list of feature and label tensors.
            batch_idx: The index of the current batch.
        """
        inputs, labels = batch
        predictions = self.forward(inputs)
        self.val_loss(predictions, labels[:, None])
        self.log("val/loss", self.val_loss)


class ConsistencyApproach(AdaptionApproach):
    """The Consistency DANN approach introduces a consistency loss that keeps the
    weights of the feature extractor close to the ones of a pre-trained version of
    itself. This approach should only be used with a pre-trained feature extractor.
    Otherwise, the consistency loss would serve no purpose.

    The regressor and domain discriminator need the same number of input units as the
    feature extractor has output units. The discriminator is not allowed to have an
    activation function on its last layer for it to work with the DANN loss.

    Examples:
        ```pycon
        >>> from rul_adapt import model
        >>> from rul_adapt import approach
        >>> feat_ex = model.CnnExtractor(1, [16, 16, 1], 10, fc_units=16)
        >>> reg = model.FullyConnectedHead(16, [1])
        >>> disc = model.FullyConnectedHead(16, [8, 1], act_func_on_last_layer=False)
        >>> pre = approach.ConsistencyApproachPretraining(0.01)
        >>> pre.set_model(feat_ex, reg, disc)
        >>> main = approach.ConsistencyApproach(1.0, 0.01)
        >>> main.set_model(pre.feature_extractor, pre.regressor, disc)
        ```
    """

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

        A frozen copy of the feature extractor is produced to be used for the
        consistency loss. The feature extractor should, therefore, be pre-trained.

        Args:
            feature_extractor: The pre-trained feature extraction network.
            regressor: The optionally pre-trained RUL regression network.
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
        """
        Return the influency of the DANN loss based on the current epoch.

        It is calculated as: `2 / (1 + math.exp(-10 * current_epoch / max_epochs)) - 1`
        """
        return 2 / (1 + math.exp(-10 * self.current_epoch / self.max_epochs)) - 1

    def configure_optimizers(self) -> torch.optim.SGD:
        """Configure an SGD optimizer to train the feature extractor, regressor and
        domain discriminator."""
        parameters = chain(
            self.feature_extractor.parameters(),
            self.regressor.parameters(),
            self.dann_loss.parameters(),
        )  # excludes frozen_feature_extractor from optimization
        optim = torch.optim.SGD(parameters, lr=self.lr)

        return optim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Predict the RUL values for a batch of input features."""
        return self.regressor(self.feature_extractor(inputs))

    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Execute one training step.

        The `batch` argument is a list of three tensors representing the source
        features, source labels and target features. Both types of features are fed
        to the feature extractor. Then the regression loss for the source domain and
        the DANN loss between domains is computed. Afterwards the consistency loss is
        calculated. The regression, DANN, consistency and combined loss are logged.

        Args:
            batch: A list of a source feature, source label and target feature tensors.
            batch_idx: The index of the current batch.
        Returns:
            The combined loss.
        """
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


class StdExtractor:
    """
    This extractor can be used to extract the per-feature standard deviation from
    windows of data. It can be used to pre-process datasets like FEMTO and XJTU-SY
    with the help of the [RulDataModule][rul_datasets.core.RulDataModule].

    Examples:
        Extract the std of the horizontal acceleration and produce windows of size 30.
        ```pycon
        >>> import rul_datasets
        >>> import rul_adapt
        >>> fd1 = rul_datasets.XjtuSyReader(fd=1)
        >>> extractor = rul_adapt.approach.consistency.StdExtractor([0])
        >>> dm = rul_datasets.RulDataModule(fd1, 32, extractor, window_size=30)
        ```
    """

    def __init__(self, channels: List[int]) -> None:
        """
        Create a new feature extractor for standard deviations.

        Args:
            channels: The list of channel indices to extract features from.
        """
        self.channels = channels

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """
        Extract features from the input data.

        The input is expected to have a shape of `[num_windows, window_size,
        num_features]`. The output will have a shape of `[num_windows,
        len(self.channels)]`.

        Args:
            inputs: The input data.
        Returns:
            The features extracted from the input data.
        """
        return np.std(inputs[:, :, self.channels], axis=1)