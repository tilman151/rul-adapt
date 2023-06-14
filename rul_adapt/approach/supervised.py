"""The supervised approach trains solely on the labeled source domain. It can be used
for pre-training or as a baseline to compare adaption approaches against.

```python
Data --> FeatureExtractor --> Features --> Regressor  --> RUL Prediction
```
"""

from typing import Tuple, Literal, Any, Dict

import torch
import torchmetrics

from rul_adapt import utils
from rul_adapt.approach.abstract import AdaptionApproach


class SupervisedApproach(AdaptionApproach):
    """The supervised approach uses either MSE, MAE or RMSE loss to train a feature
    extractor and regressor in a supervised fashion on the source domain. It can be
    used either for pre-training or as a baseline to compare adaption approaches
    against.

    The regressor needs the same number of input units as the feature extractor has
    output units.

    Examples:
        >>> from rul_adapt import model
        >>> from rul_adapt import approach
        >>> feat_ex = model.CnnExtractor(1, [16, 16, 1], 10, fc_units=16)
        >>> reg = model.FullyConnectedHead(16, [1])
        >>> disc = model.FullyConnectedHead(16, [8, 1], act_func_on_last_layer=False)
        >>> main = approach.SupervisedApproach("mse")
        >>> main.set_model(feat_ex, reg, disc)
    """

    def __init__(
        self,
        loss_type: Literal["mse", "mae", "rmse"],
        rul_scale: int = 1,
        **optim_kwargs: Any,
    ) -> None:
        """
        Create a supervised approach.

        The regressor output can be scaled with `rul_scale` to control its
        magnitude. By default, the RUL values are not scaled.

        For more information about the possible optimizer keyword arguments,
        see [here][rul_adapt.utils.OptimizerFactory].

        Args:
            loss_type: Training loss function to use. Either 'mse', 'mae' or 'rmse'.
            rul_scale: Scalar to multiply the RUL prediction with.
            **optim_kwargs: Keyword arguments for the optimizer, e.g. learning rate.
        """
        super().__init__()

        self.loss_type = loss_type
        self.rul_scale = rul_scale
        self.optim_kwargs = optim_kwargs

        self.train_loss = utils.get_loss(loss_type)
        self._get_optimizer = utils.OptimizerFactory(**self.optim_kwargs)
        self.val_loss = torchmetrics.MeanSquaredError(squared=False)

        self.save_hyperparameters()

    def configure_optimizers(self) -> Dict[str, Any]:
        return self._get_optimizer(self.parameters())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.feature_extractor(inputs)) * self.rul_scale

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Execute one training step.

        The `batch` argument is a list of two tensors representing features and
        labels. The features are used to predict RUL values that are compared against
        the labels with the specified training loss. The loss is then logged.

        Args:
            batch: A list of feature and label tensors.
            batch_idx: The index of the current batch.
        Returns:
            The training loss.
        """
        inputs, labels = batch
        predictions = self.forward(inputs)
        loss = self.train_loss(predictions, labels[:, None])
        self.log("train/loss", self.train_loss)

        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
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
