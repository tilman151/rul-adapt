"""The Domain Adversarial Neural Network (DANN) approach uses a domain discriminator
trained on distinguishing the source and target features produced by a shared feature
extractor. A [Gradient Reversal Layer][rul_adapt.loss.adaption.GradientReversalLayer]
(GRL) is used to train the feature extractor on making its source and target outputs
indistinguishable.

```python
Source --> FeatEx --> Source Feats -----------> Regressor  --> RUL Prediction
        ^         |                 |
        |         |                 v
Target --         --> Target Feats -->  GRL --> DomainDisc --> Domain Prediction
```

It was originally introduced by [Ganin et al.](http://jmlr.org/papers/v17/15-239.html)
for image classification.

Used In:
    * da Costa et al. (2020). **Remaining useful lifetime prediction via deep domain
    adaptation.**
    *Reliability Engineering & System Safety*, *195*, 106682.
    [10.1016/J.RESS.2019.106682](https://doi.org/10.1016/J.RESS.2019.106682)
    * Krokotsch et al. (2020). **A Novel Evaluation Framework for Unsupervised Domain
    Adaption on Remaining Useful Lifetime Estimation.**
    *2020 IEEE International Conference on Prognostics and Health Management (ICPHM)*.
    [10.1109/ICPHM49022.2020.9187058](https://doi.org/10.1109/ICPHM49022.2020.9187058)
"""

from typing import Any, Optional, Dict, Literal, List

import torch
from torch import nn

import rul_adapt.loss
from rul_adapt import utils
from rul_adapt.approach.abstract import AdaptionApproach
from rul_adapt.approach.evaluation import AdaptionEvaluator
from rul_adapt.model import FullyConnectedHead


class DannApproach(AdaptionApproach):
    """The DANN approach introduces a domain discriminator that is trained on
    distinguishing source and target features as a binary classification problem. The
    features are produced by a shared feature extractor. The loss in the domain
    discriminator is binary cross-entropy.

    The regressor and domain discriminator need the same number of input units as the
    feature extractor has output units. The discriminator is not allowed to have an
    activation function on its last layer and needs to use only a single output
    neuron because [BCEWithLogitsLoss][torch.nn.BCEWithLogitsLoss] is used.

    Examples:
        >>> from rul_adapt import model
        >>> from rul_adapt import approach
        >>> feat_ex = model.CnnExtractor(1, [16, 16, 1], 10, fc_units=16)
        >>> reg = model.FullyConnectedHead(16, [1])
        >>> disc = model.FullyConnectedHead(16, [8, 1], act_func_on_last_layer=False)
        >>> dann = approach.DannApproach(1.0)
        >>> dann.set_model(feat_ex, reg, disc)
    """

    CHECKPOINT_MODELS = ["dann_loss"]

    dann_loss: rul_adapt.loss.DomainAdversarialLoss

    def __init__(
        self,
        dann_factor: float,
        loss_type: Literal["mae", "mse", "rmse"] = "mae",
        rul_score_mode: Literal["phm08", "phm12"] = "phm08",
        evaluate_degraded_only: bool = False,
        **optim_kwargs: Any,
    ):
        """
        Create a new DANN approach.

        The strength of the domain discriminator's influence on the feature
        extractor is controlled by the `dann_factor`. The higher it is, the stronger
        the influence.

        Possible options for the regression loss are `mae`, `mse` and `rmse`.

        The domain discriminator is set by the `set_model` function together with the
        feature extractor and regressor. For more information, see the [approach]
        [rul_adapt.approach] module page.

        For more information about the possible optimizer keyword arguments,
        see [here][rul_adapt.utils.OptimizerFactory].

        Args:
            dann_factor: Strength of the domain DANN loss.
            loss_type: Type of regression loss.
            rul_score_mode: The mode for the val and test RUL score, either 'phm08'
                            or 'phm12'.
            evaluate_degraded_only: Whether to only evaluate the RUL score on degraded
                                    samples.
            **optim_kwargs: Keyword arguments for the optimizer, e.g. learning rate.
        """
        super().__init__()

        self.dann_factor = dann_factor
        self.loss_type = loss_type
        self.rul_score_mode = rul_score_mode
        self.evaluate_degraded_only = evaluate_degraded_only
        self.optim_kwargs = optim_kwargs

        self._get_optimizer = utils.OptimizerFactory(**self.optim_kwargs)

        self.train_source_loss = utils.get_loss(self.loss_type)
        self.evaluator = AdaptionEvaluator(
            self.forward, self.log, self.rul_score_mode, self.evaluate_degraded_only
        )

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
        Set the feature extractor, regressor, and domain discriminator for this
        approach.

        The discriminator is not allowed to have an activation function on its last
        layer and needs to use only a single output neuron. It is wrapped by a
        [DomainAdversarialLoss][rul_adapt.loss.DomainAdversarialLoss].

        Args:
            feature_extractor: The feature extraction network.
            regressor: The RUL regression network.
            domain_disc: The domain discriminator network.
        """
        domain_disc = self._check_domain_disc(domain_disc)
        super().set_model(feature_extractor, regressor, *args, **kwargs)
        self.dann_loss = rul_adapt.loss.DomainAdversarialLoss(domain_disc)
        self.log_model_hyperparameters("domain_disc")

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

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure an optimizer for the whole model."""
        return self._get_optimizer(self.parameters())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Predict the RUL values for a batch of input features."""
        return self.regressor(self.feature_extractor(inputs))

    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Execute one training step.

        The `batch` argument is a list of three tensors representing the source
        features, source labels and target features. Both types of features are fed
        to the feature extractor. Then the regression loss for the source domain and
        the DANN loss between domains is computed. The regression, DANN and combined
        loss are logged.

        Args:
            batch: A list of a source feature, source label and target feature tensors.
            batch_idx: The index of the current batch.
        Returns:
            The combined loss.
        """
        source, source_labels, target = batch
        source_labels = source_labels[:, None]

        source = self.feature_extractor(source)
        target = self.feature_extractor(target)

        rul_preds = self.regressor(source)
        mse_loss = self.train_source_loss(rul_preds, source_labels)

        dann_loss = self.dann_loss(source, target)

        loss = mse_loss + self.dann_factor * dann_loss

        self.log("train/loss", loss)
        self.log("train/source_loss", self.train_source_loss)
        self.log("train/dann", self.dann_loss)

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
