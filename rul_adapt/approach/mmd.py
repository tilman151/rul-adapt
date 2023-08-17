"""The Maximum Mean Discrepancy (MMD) approach uses the distance measure of the same
name to adapt a feature extractor. This implementation uses a multi-kernel variant of
the [MMD loss][rul_adapt.loss.adaption.MaximumMeanDiscrepancyLoss] with bandwidths
set via the median heuristic.

```python
Source --> FeatEx --> Source Feats -----------> Regressor  --> RUL Prediction
        ^         |                 |
        |         |                 v
Target --         --> Target Feats -->  MMD Loss
```

It was first introduced by [Long et al.](
https://dl.acm.org/doi/10.5555/3045118.3045130) as Deep Adaption Network (DAN) for
image classification.

Used In:
    * Cao et al. (2021). **Transfer learning for remaining useful life prediction of
    multi-conditions bearings based on bidirectional-GRU network.**
    *Measurement: Journal of the International Measurement Confederation*, *178*.
    [10.1016/j.measurement.2021.109287](https://doi.org/10.1016/j.measurement.2021.109287)
    * Krokotsch et al. (2020). **A Novel Evaluation Framework for Unsupervised Domain
    Adaption on Remaining Useful Lifetime Estimation.**
    *2020 IEEE International Conference on Prognostics and Health Management (ICPHM)*.
    [10.1109/ICPHM49022.2020.9187058](https://doi.org/10.1109/ICPHM49022.2020.9187058)
"""

from typing import List, Literal, Any, Dict

import torch

import rul_adapt
from rul_adapt import utils
from rul_adapt.approach.abstract import AdaptionApproach
from rul_adapt.approach.evaluation import AdaptionEvaluator


class MmdApproach(AdaptionApproach):
    """The MMD uses the Maximum Mean Discrepancy to adapt a feature extractor to
    be used with the source regressor.

    The regressor needs the same number of input units as the feature extractor has
    output units.

    Examples:
        >>> from rul_adapt import model
        >>> from rul_adapt import approach
        >>> feat_ex = model.CnnExtractor(1, [16, 16, 1], 10, fc_units=16)
        >>> reg = model.FullyConnectedHead(16, [1])
        >>> mmd = approach.MmdApproach(0.01)
        >>> mmd.set_model(feat_ex, reg)
    """

    def __init__(
        self,
        mmd_factor: float,
        num_mmd_kernels: int = 5,
        loss_type: Literal["mse", "rmse", "mae"] = "mse",
        rul_score_mode: Literal["phm08", "phm12"] = "phm08",
        evaluate_degraded_only: bool = False,
        **optim_kwargs: Any,
    ) -> None:
        """
        Create a new MMD approach.

        The strength of the influence of the MMD loss on the feature
        extractor is controlled by the `mmd_factor`. The higher it is, the stronger
        the influence.

        For more information about the possible optimizer keyword arguments,
        see [here][rul_adapt.utils.OptimizerFactory].

        Args:
            mmd_factor: The strength of the MMD loss' influence.
            num_mmd_kernels: The number of kernels for the MMD loss.
            loss_type: The type of regression loss, either 'mse', 'rmse' or 'mae'.
            rul_score_mode: The mode for the val and test RUL score, either 'phm08'
                            or 'phm12'.
            evaluate_degraded_only: Whether to only evaluate the RUL score on degraded
                                    samples.
            **optim_kwargs: Keyword arguments for the optimizer, e.g. learning rate.
        """
        super().__init__()

        self.mmd_factor = mmd_factor
        self.num_mmd_kernels = num_mmd_kernels
        self.loss_type = loss_type
        self.rul_score_mode = rul_score_mode
        self.evaluate_degraded_only = evaluate_degraded_only
        self.optim_kwargs = optim_kwargs

        # training metrics
        self.train_source_loss = utils.get_loss(self.loss_type)
        self.mmd_loss = rul_adapt.loss.MaximumMeanDiscrepancyLoss(self.num_mmd_kernels)
        self._get_optimizer = utils.OptimizerFactory(**self.optim_kwargs)
        self.evaluator = AdaptionEvaluator(
            self.forward, self.log, self.rul_score_mode, self.evaluate_degraded_only
        )

        self.save_hyperparameters()

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure an optimizer."""
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
        the MMD loss between domains is computed. The regression, MMD and combined
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
        mmd_loss = self.mmd_loss(source, target)
        loss = mse_loss + self.mmd_factor * mmd_loss

        self.log("train/loss", loss)
        self.log("train/source_loss", self.train_source_loss)
        self.log("train/mmd", self.mmd_loss)

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
