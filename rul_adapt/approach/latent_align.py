"""The latent space alignment approach uses several auxiliary losses to align the
latent space of the source and target domain produced by a shared feature extractor:

* **Healthy State Alignment:** Pushes the healthy data of both domains into a single
  compact cluster
* **Degradation Direction Alignment:** Minimizes the angle between degraded data
  points with the healthy cluster as origin
* **Degradation Level Alignment:** Aligns the distance of degraded data points from the
  healthy cluster to the number of time steps in degradation
* **Degradation Fusion:** Uses a
  [MMD loss][rul_adapt.loss.adaption.MaximumMeanDiscrepancyLoss] to align the
  distribution of both domains

Which features are considered in the healthy state and which in degradation is either
determined by taking the first few steps of each time series or by using a
first-time-to-predict estimation. The first variant is used for CMAPSS, the second
for XJTU-SY.

The approach was introduced by [Zhang et al.](
https://doi.org/10.1016/j.ress.2021.107556) in 2021. For applying the approach on raw
vibration data, i.e. XJTU-SY, it uses a [windowing scheme]
[rul_adapt.approach.latent_align.extract_chunk_windows] and
[first-point-to-predict estimation]
[rul_adapt.approach.latent_align.LatentAlignFttpApproach] introduced by [Li et al.](
https://doi.org/10.1016/j.knosys.2020.105843) in 2020."""

from typing import Tuple, List, Any, Optional, Literal, Dict

import numpy as np
import torch
from rul_datasets.utils import feature_to_tensor
from torch import nn

import rul_adapt
from rul_adapt import utils
from rul_adapt.approach.abstract import AdaptionApproach
from rul_adapt.approach.evaluation import AdaptionEvaluator


class LatentAlignFttpApproach(AdaptionApproach):
    """This first-point-to-predict estimation approach trains a GAN on healthy state
    bearing data. The discriminator can be used afterward to compute a health
    indicator for each bearing.

    The feature extractor and regressor models are used as the discriminator. The
    regressor is not allowed to have an activation function on its last layer and
    needs to use only a single output neuron because [BCEWithLogitsLoss]
    [torch.nn.BCEWithLogitsLoss] is used. The generator receives noise with the shape
    [batch_size, 1, noise_dim]. The generator needs an output with enough elements so
    that it can be reshaped to the same shape as the real input data. The reshaping
    is done internally.

    Both generator and discriminator are trained at once by using a
    [Gradient Reversal Layer][rul_adapt.loss.adaption.GradientReversalLayer]
    between them.

    Examples:
        >>> from rul_adapt import model, approach
        >>> feat_ex = model.CnnExtractor(1, [16, 16, 1], 10, fc_units=16)
        >>> reg = model.FullyConnectedHead(16, [1])
        >>> gen = model.CnnExtractor(1, [1], 10, padding=True)
        >>> fttp_model = approach.LatentAlignFttpApproach(1e-4, 10)
        >>> fttp_model.set_model(feat_ex, reg, gen)
        >>> health_indicator = fttp_model(torch.randn(16, 1, 10)).std()

    """

    CHECKPOINT_MODELS = ["_generator"]

    _generator: nn.Module

    def __init__(
        self,
        noise_dim: int,
        **optim_kwargs: Any,
    ):
        """
        Create a new FTTP estimation approach.

        The generator is set by the `set_model` function together with the feature
        extractor and regressor.

        For more information about the possible optimizer keyword arguments,
        see [here][rul_adapt.utils.OptimizerFactory].

        Args:
            noise_dim: The size of the last dimension of the noise tensor.
            **optim_kwargs: Keyword arguments for the optimizer, e.g. learning rate.
        """
        super().__init__()

        self.noise_dim = noise_dim
        self.optim_kwargs = optim_kwargs

        self.gan_loss = torch.nn.BCEWithLogitsLoss()
        self.grl = rul_adapt.loss.adaption.GradientReversalLayer()
        self._get_optimizer = utils.OptimizerFactory(**self.optim_kwargs)

        self.save_hyperparameters()

    def set_model(
        self,
        feature_extractor: nn.Module,
        regressor: nn.Module,
        generator: Optional[nn.Module] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Set the feature extractor, regressor (forming the discriminator) and
        generator for this approach.

        The regressor is not allowed to have an activation function on its last layer
        and needs to use only a single output neuron. The generator receives noise
        with the shape [batch_size, 1, noise_dim]. The generator needs an output with
        enough elements so that it can be reshaped to the same shape as the real
        input data. The reshaping is done internally.

        Args:
            feature_extractor: The feature extraction network.
            regressor: The regressor functioning as the head of the discriminator.
            generator: The generator network.
        """
        super().set_model(feature_extractor, regressor)
        if generator is None:
            raise ValueError("Generator not set. This approach is unlikely to work.")
        self._generator = generator

    @property
    def generator(self):
        """The generator network."""
        if hasattr(self, "_generator"):
            return self._generator
        else:
            raise RuntimeError("Generator used before 'set_model' was called.")

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure an optimizer for the generator and discriminator."""
        return self._get_optimizer(self.parameters())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Predict the health indicator for the given inputs."""
        return self.regressor(self.feature_extractor(inputs))

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Execute one training step.

        The batch is a tuple of the features and the labels. The labels are ignored.
        A noise tensor is passed to the generator to generate fake features. The
        discriminator classifies if the features are real or fake and the binary
        cross entropy loss is calculated. Real features receive the label zero and
        the fake features one.

        Both generator and discriminator are trained at once by using a
        [Gradient Reversal Layer][rul_adapt.loss.adaption.GradientReversalLayer]
        between them. At the end, the loss is logged.

        Args:
            batch: A tuple of feature and label tensors.

        Returns:
            The classification loss.
        """
        features, _ = batch
        batch_size = features.shape[0]
        device: torch.device = self.device  # type: ignore[assignment]

        pred_real = self.forward(features)
        loss_real = self.gan_loss(pred_real, torch.zeros(batch_size, 1, device=device))

        noise = torch.randn(batch_size, 1, self.noise_dim, device=device)
        fake_features = self.grl(self.generator(noise)).reshape_as(features)
        pred_fake = self.forward(fake_features)
        loss_fake = self.gan_loss(pred_fake, torch.ones(batch_size, 1, device=device))

        loss = (loss_real + loss_fake) / 2
        self.log("train/loss", loss)

        return loss


def get_first_time_to_predict(
    fttp_model: LatentAlignFttpApproach,
    features: np.ndarray,
    window_size: int,
    chunk_size: int,
    healthy_index: int,
    threshold_coefficient: float,
) -> int:
    """
    Get the first time step to predict for the given features.

    The features are pre-processed via the [extract_chunk_windows]
    [rul_adapt.approach.latent_align.extract_chunk_windows] function and fed in
    batches to the `fttp_model`. Each batch consists of the chunk windows that end in
    the same original feature window. The health indicator for the original window is
    calculated as the standard deviation of the predictions of the `fttp_model`.

    The first-time-to-predict is the first time step where the health indicator is
    larger than `threshold_coefficient` times the mean of the health indicator for
    the first `healthy_index` time steps. If the threshold is never exceeded,
    a RuntimeError is raised.

    Args:
        fttp_model: The model to use for the health indicator calculation.
        features: The features to calculate the first-time-to-predict for.
        window_size: The size of the chunk windows to extract.
        chunk_size: The size of the chunks for each chunk window to extract.
        healthy_index: The index of the last healthy time step.
        threshold_coefficient: The threshold coefficient for the health indicator.

    Returns:
        The original window index of the first-time-to-predict.
    """
    if threshold_coefficient <= 1:
        raise ValueError("Threshold coefficient needs to be greater than one.")

    health_indicator = get_health_indicator(
        fttp_model, features, window_size, chunk_size
    )
    offset = len(features) - len(health_indicator)  # windowing cuts off first windows
    healthy = np.mean(health_indicator[: healthy_index - offset])
    over_thresh = np.argwhere(health_indicator > threshold_coefficient * healthy)

    if len(over_thresh) == 0:
        raise RuntimeError("Health indicator never passes threshold.")
    fttp = over_thresh[0, 0] + offset

    return fttp


@torch.no_grad()
def get_health_indicator(
    fttp_model: nn.Module, features: np.ndarray, window_size: int, chunk_size: int
) -> np.ndarray:
    """
    Get the health indicator for the given features.

    The features are pre-processed via the [extract_chunk_windows]
    [rul_adapt.approach.latent_align.extract_chunk_windows] function and fed in
    batches to the `fttp_model`. Each batch consists of the chunk windows that end in
    the same original feature window. The health indicator for the original window is
    calculated as the standard deviation of the predictions of the `fttp_model`.

    The length of the returned health indicator array is shorter than the `features`
    array by `window_size - 1`, due to the chunk windowing. This means the first
    health indicator value belongs to the original window with the index
    `window_size - 1`.

    Args:
        fttp_model: The model to use for the health indicator calculation.
        features: The features to calculate the health indicator for.
        window_size: The size of the chunk windows to extract.
        chunk_size: The size of the chunks for each chunk window to extract.

    Returns:
        The health indicator for the original windows.
    """
    chunked = extract_chunk_windows(features, window_size, chunk_size)
    chunks_per_window = features.shape[1] // chunk_size
    batches = np.split(chunked, len(chunked) // chunks_per_window)
    health_indicator = np.empty(len(batches))
    for i, batch in enumerate(batches):
        preds = fttp_model(feature_to_tensor(batch, torch.float))
        health_indicator[i] = np.std(preds.detach().numpy())

    return health_indicator


def extract_chunk_windows(
    features: np.ndarray, window_size: int, chunk_size: int
) -> np.ndarray:
    """
    Extract chunk windows from the given features of shape `[num_org_windows,
    org_window_size, num_features]`.

    A chunk window is a window that consists of `window_size` chunks. Each original
    window is split into chunks of size `chunk_size`. A chunk window is then formed
    by concatenating chunks from the same position inside `window_size` consecutive
    original windows. Therefore, each original window is represented by
    `org_window_size // chunk_size` chunk windows. The original window size must
    therefor be divisible by the chunk size.

    Args:
        features: The features to extract the chunk windows from.
        window_size: The number of consecutive original windows to form a chunk window
                     from.
        chunk_size: The size of the chunks to extract from the original windows.

    Returns:
        Chunk windows of shape `[num_windows, window_size * chunk_size, num_features]`.
    """
    old_window_size = features.shape[1]
    window_multiplier = old_window_size // chunk_size
    num_new_windows = (features.shape[0] - window_size + 1) * window_multiplier

    chunk_idx = np.tile(np.arange(chunk_size), num_new_windows * window_size)
    intra_offsets = np.tile(np.arange(window_size), num_new_windows) * old_window_size
    inter_offsets = np.repeat(np.arange(num_new_windows), window_size) * chunk_size
    offsets = np.repeat(intra_offsets + inter_offsets, chunk_size)
    window_idx = chunk_idx + offsets

    flat_features = features.reshape((-1, features.shape[2]))
    flat_windows = flat_features[window_idx]
    windows = flat_windows.reshape((num_new_windows, window_size * chunk_size, -1))

    return windows


class ChunkWindowExtractor:
    def __init__(self, window_size: int, chunk_size: int) -> None:
        self.window_size = window_size
        self.chunk_size = chunk_size

    def __call__(
        self, features: np.ndarray, targets: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        chunks_per_window = features.shape[1] // self.chunk_size
        chunked = extract_chunk_windows(features, self.window_size, self.chunk_size)
        targets = targets[self.window_size - 1 :].repeat(chunks_per_window)

        return chunked, targets


class LatentAlignApproach(AdaptionApproach):
    """
    The latent alignment approach introduces four latent space alignment losses to
    align the latent space of a shared feature extractor to both source and target
    domain.

    Examples:
        >>> from rul_adapt import model, approach
        >>> feat_ex = model.CnnExtractor(1, [16, 16, 1], 10, fc_units=16)
        >>> reg = model.FullyConnectedHead(16, [1])
        >>> latent_align = approach.LatentAlignApproach(0.1, 0.1, 0.1, 0.1, lr=0.001)
        >>> latent_align.set_model(feat_ex, reg)
    """

    def __init__(
        self,
        alpha_healthy: float,
        alpha_direction: float,
        alpha_level: float,
        alpha_fusion: float,
        loss_type: Literal["mse", "mae", "rmse"] = "mse",
        rul_score_mode: Literal["phm08", "phm12"] = "phm08",
        evaluate_degraded_only: bool = False,
        labels_as_percentage: bool = False,
        **optim_kwargs: Any,
    ) -> None:
        """
        Create a new latent alignment approach.

        Each of the alphas controls the influence of the respective loss on the
        training. Commonly they are all set to the same value.

        For more information about the possible optimizer keyword arguments,
        see [here][rul_adapt.utils.OptimizerFactory].

        Args:
            alpha_healthy: The influence of the healthy state alignment loss.
            alpha_direction: The influence of the degradation direction alignment loss.
            alpha_level: The influence of the degradation level regularization loss.
            alpha_fusion: The influence of the degradation fusion (MMD) loss.
            loss_type: The type of regression loss to use.
            rul_score_mode: The mode for the val and test RUL score, either 'phm08'
                            or 'phm12'.
            evaluate_degraded_only: Whether to only evaluate the RUL score on degraded
                                    samples.
            labels_as_percentage: Whether to multiply labels by 100 to get percentages
            **optim_kwargs: Keyword arguments for the optimizer, e.g. learning rate.
        """
        super().__init__()

        self.alpha_healthy = alpha_healthy
        self.alpha_direction = alpha_direction
        self.alpha_level = alpha_level
        self.alpha_fusion = alpha_fusion
        self.loss_type = loss_type
        self.rul_score_mode = rul_score_mode
        self.evaluate_degraded_only = evaluate_degraded_only
        self.labels_as_percentage = labels_as_percentage
        self.optim_kwargs = optim_kwargs

        # training metrics
        self.train_mse = utils.get_loss(self.loss_type)
        self.healthy_align = rul_adapt.loss.HealthyStateAlignmentLoss()
        self.direction_align = rul_adapt.loss.DegradationDirectionAlignmentLoss()
        self.level_align = rul_adapt.loss.DegradationLevelRegularizationLoss()
        self.fusion_align = rul_adapt.loss.MaximumMeanDiscrepancyLoss(num_kernels=5)
        self._get_optimizer = utils.OptimizerFactory(**self.optim_kwargs)

        self.evaluator = AdaptionEvaluator(
            self.forward, self.log, self.rul_score_mode, self.evaluate_degraded_only
        )

        self.save_hyperparameters()

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure an optimizer."""
        optim = self._get_optimizer(self.parameters())

        return optim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict the RUL values for a batch of input features."""
        preds = self.regressor(self.feature_extractor(features))
        if self.labels_as_percentage:
            preds = self._from_percentage(preds)

        return preds

    def training_step(
        self, batch: Tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor:
        """
        Execute one training step.

        The `batch` contains the following tensors in order:

        * The source domain features.
        * The steps in degradation for the source features.
        * The RUL labels for the source features.
        * The target domain features.
        * The steps in degradation for the target features.
        * The healthy state features for both domains.

        The easies way to produce such a batch is using the [LatentAlignDataModule]
        [rul_datasets.adaption.LatentAlignDataModule].

        The source, target and healthy features are passed through the feature
        extractor. Afterward, these high-level features are used to compute the
        alignment losses. The source domain RUL predictions are computed using the
        regressor and used to calculate the MSE loss. The losses are then combined.
        Each separate and the combined loss are logged.

        Args:
            batch: The batch of data.
            batch_idx: The index of the batch.

        Returns:
            The combined loss.
        """
        source, source_degradation_steps, source_labels, *_ = batch
        *_, target, target_degradation_steps, healthy = batch
        if self.labels_as_percentage:
            source_labels = self._to_percentage(source_labels)
        else:
            source_labels = source_labels[:, None]

        healthy = self.feature_extractor(healthy)
        source = self.feature_extractor(source)
        target = self.feature_extractor(target)

        rul_predictions = self.regressor(source)
        mse_loss = self.train_mse(rul_predictions, source_labels)

        healthy_loss = self.healthy_align(healthy)
        direction_loss = self.direction_align(healthy, torch.cat([source, target]))
        level_loss = self.level_align(
            healthy, source, source_degradation_steps, target, target_degradation_steps
        )
        fusion_loss = self.fusion_align(source, target)

        loss = (
            mse_loss
            + self.alpha_healthy * healthy_loss
            + self.alpha_direction * direction_loss
            + self.alpha_level * level_loss
            + self.alpha_fusion * fusion_loss
        )

        self.log("train/loss", loss)
        self.log("train/mse", self.train_mse)
        self.log("train/healthy_align", self.healthy_align)
        self.log("train/direction_align", self.direction_align)
        self.log("train/level_align", self.level_align)
        self.log("train/fusion_align", self.fusion_align)

        return loss

    def _to_percentage(self, source_labels):
        """Convert RUL labels to percentages assuming they are normed between [0, 1]."""
        return source_labels[:, None] * 100

    def _from_percentage(self, predictions: torch.Tensor) -> torch.Tensor:
        return predictions / 100

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
