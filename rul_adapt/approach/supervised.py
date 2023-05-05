from typing import Tuple, Literal

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
        ```pycon
        >>> from rul_adapt import model
        >>> from rul_adapt import approach
        >>> feat_ex = model.CnnExtractor(1, [16, 16, 1], 10, fc_units=16)
        >>> reg = model.FullyConnectedHead(16, [1])
        >>> disc = model.FullyConnectedHead(16, [8, 1], act_func_on_last_layer=False)
        >>> pre = approach.SupervisedApproach(0.01, "mse", 125)
        >>> pre.set_model(feat_ex, reg)
        >>> main = approach.SupervisedApproach(0.01,,125
        >>> main.set_model(pre.feature_extractor, pre.regressor, disc)
        ```
    """

    def __init__(
        self,
        lr: float,
        loss_type: Literal["mse", "mae", "rmse"],
        optim_type: Literal["adam", "sgd"],
        rul_scale: int = 1,
    ) -> None:
        """
        Create a supervised approach.

        The regressor is supposed to output a value between [0, 1] which is then
        scaled by `rul_scale`. By default, the RUL values are not scaled.

        Args:
            lr: Learning rate.
            rul_scale: Scalar to multiply the RUL prediction with.
            optim_type: Optimizer to use. Either 'adam' or 'sgd'.
            loss_type: Training loss function to use. Either 'mse', 'mae' or 'rmse'.
        """
        super().__init__()

        self.lr = lr
        self.loss_type = loss_type
        self.optim_type = optim_type
        self.rul_scale = rul_scale

        self.train_loss = utils.get_loss(loss_type)
        self.val_loss = torchmetrics.MeanSquaredError(squared=False)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optim: torch.optim.Optimizer
        if self.optim_type == "adam":
            optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optim_type == "sgd":
            optim = torch.optim.SGD(self.parameters(), lr=self.lr)
        else:
            raise ValueError(
                f"Unknown optimizer '{self.optim_type}'. " "Use either 'adam' or 'sgd'."
            )

        return optim

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
