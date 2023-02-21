from itertools import product
from typing import List, Tuple, Optional

import numpy as np
import pywt  # type: ignore
import torch
import torchmetrics
from dtaidistance import dtw  # type: ignore
from rul_datasets.reader import AbstractReader
from rul_datasets.utils import extract_windows
from scipy.stats import wasserstein_distance  # type: ignore
from sklearn.preprocessing import MinMaxScaler  # type: ignore

import rul_adapt
from rul_adapt import utils
from rul_adapt.approach.abstract import AdaptionApproach
from rul_adapt.approach.modwt import modwt


def rms(inputs: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(inputs**2, axis=-2))


def p2p(inputs: np.ndarray) -> np.ndarray:
    return np.max(inputs, axis=-2) - np.min(inputs, axis=-2)


def skewness(inputs: np.ndarray) -> np.ndarray:
    sdev = inputs.std(axis=-2)
    mean = inputs.mean(axis=-2, keepdims=True)
    skew = np.sum((inputs - mean) ** 3, axis=-2) / (inputs.shape[-2] * sdev**3)

    return skew


def impulse_factor(inputs: np.ndarray) -> np.ndarray:
    absolute = np.abs(inputs)
    imp_f = np.max(absolute, axis=-2) / np.mean(absolute, axis=-2)

    return imp_f


def median_absolute(inputs: np.ndarray) -> np.ndarray:
    return np.median(np.abs(inputs), axis=-2)


def mean_absolute(inputs: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(inputs), axis=-2)


def max_absolute(inputs: np.ndarray) -> np.ndarray:
    return np.max(np.abs(inputs), axis=-2)


def kurtosis(inputs: np.ndarray) -> np.ndarray:
    sdev = inputs.std()
    mean = inputs.mean(axis=-2, keepdims=True)
    kurt = np.sum((inputs - mean) ** 4, axis=-2) / (inputs.shape[-2] * sdev**2)

    return kurt


def std(inputs: np.ndarray) -> np.ndarray:
    return np.std(inputs, axis=-2)


def margin_factor(inputs: np.ndarray) -> np.ndarray:
    absolute = np.abs(inputs)
    mf = np.max(absolute, axis=-2) / (np.mean(np.sqrt(absolute), axis=-2)) ** 2

    return mf


def energy(inputs: np.ndarray) -> np.ndarray:
    return np.sum(inputs**2, axis=-2)


def gini_factor(inputs: np.ndarray) -> np.ndarray:
    batched = len(inputs.shape) == 3
    inputs = inputs if batched else inputs[None]
    gini = _approx_batched_gini_factor(inputs)
    gini = gini if batched else gini.squeeze(0)

    return gini


def _approx_batched_gini_factor(inputs: np.ndarray) -> np.ndarray:
    window_size = inputs.shape[1]
    inputs = np.sort(inputs, axis=1)
    cumsum = np.cumsum(inputs, axis=1)
    gini = (window_size + 1 - 2 * np.sum(cumsum, axis=1) / cumsum[:, -1]) / window_size

    return gini


def band_energies(inputs: np.ndarray) -> np.ndarray:
    wp = pywt.WaveletPacket(inputs, wavelet="db1", maxlevel=4, axis=-2)
    bands = (node.data for node in wp.get_level(4))
    energies = np.concatenate([energy(b) for b in bands], axis=-1)

    return energies


def std_ihc(inputs: np.ndarray) -> np.ndarray:
    return np.std((np.arccosh(inputs + 0j)), axis=-2)


def std_ihs(inputs: np.ndarray) -> np.ndarray:
    return np.std(np.arcsinh(inputs), axis=-2)


class VibrationFeatureExtractor:

    _scaler: Optional[MinMaxScaler]

    def __init__(
        self, num_input_features: int, feature_idx: Optional[List[int]] = None
    ) -> None:
        self.num_input_features = num_input_features
        self.feature_idx = list(range(60)) if feature_idx is None else feature_idx

        if min(self.feature_idx) < 0 or max(self.feature_idx) > 60:
            raise ValueError("Feature indices need to be between 0 and 60.")

        self._scaler = None

    def __call__(
        self, features: np.ndarray, targets: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        features = self._extract_selected(features)
        if self._scaler is not None:
            features = self._scaler.transform(features)

        return features, targets

    def _extract_selected(self, features: np.ndarray) -> np.ndarray:
        return _extract_all(features, self.num_input_features)[:, self.feature_idx]

    def fit(self, features: List[np.ndarray]) -> "VibrationFeatureExtractor":
        features = [self._extract_selected(f) for f in features]
        self._scaler = MinMaxScaler()
        for feat in features:
            self._scaler.partial_fit(feat)

        return self


def _extract_all(features: np.ndarray, num_features: int) -> np.ndarray:
    feature_list = [
        rms(features),
        kurtosis(features),
        p2p(features),
        std(features),
        skewness(features),
        margin_factor(features),
        impulse_factor(features),
        energy(features),
        median_absolute(features),
        gini_factor(features),
        max_absolute(features),
        mean_absolute(features),
        *band_energies(features)
        .reshape((-1, 16, num_features))
        .transpose(1, 0, 2),  # unpack features
        std_ihc(features),
        std_ihs(features),
    ]
    features = np.concatenate(feature_list, axis=1)

    return features


def domain_distance(source: np.ndarray, target: np.ndarray, ratio=0.4) -> float:
    dtw_dist = dtw.distance_fast(source, target)
    wasserstein_dist = wasserstein_distance(source, target)
    distance = ratio * dtw_dist + (1 - ratio) * wasserstein_dist

    return distance


def select_features(
    source: AbstractReader, target: AbstractReader, num_features: int
) -> List[int]:
    source.prepare_data()
    target.prepare_data()
    source_runs, _ = source.load_split("dev")
    target_runs, _ = target.load_split("dev")

    num_org_features = source_runs[0].shape[-1]
    distances = np.zeros(30 * num_org_features)
    for source_run, target_run in product(source_runs, target_runs):
        source_feats = _extract_all(source_run, num_features=num_org_features)
        target_feats = _extract_all(target_run, num_features=num_org_features)
        for i, (s, f) in enumerate(zip(source_feats.T, target_feats.T)):
            distances[i] += domain_distance(s, f)

    feature_idx = np.argsort(distances)[:num_features].tolist()

    return feature_idx


def mac(inputs: np.ndarray, window_size: int, wavelet: str = "sym4") -> np.ndarray:
    entropies = energy_entropies(inputs, wavelet)
    entropies = extract_windows(entropies, window_size)
    anchor, queries = entropies[:, -2:-1], entropies[:, :-1]
    corr = pearson(anchor, queries)
    corr = np.mean(np.abs(corr), axis=1)

    return corr


def energy_entropies(inputs: np.ndarray, wavelet: str = "sym4") -> np.ndarray:
    coeffs = modwt(inputs, wavelet, 4)
    energies = energy(coeffs)
    ratios = energies / np.sum(energies, axis=-1, keepdims=True)
    entropy = -np.sum(ratios * np.log(ratios), axis=-1, keepdims=True)
    entropies = ratios / entropy

    return entropies


def pearson(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    diff_x = x - np.mean(x, axis=2, keepdims=True)
    diff_y = y - np.mean(y, axis=2, keepdims=True)
    cov = np.sum(diff_y * diff_x, axis=2)
    std_product = np.sqrt(np.sum(diff_y**2, axis=2) * np.sum(diff_x**2, axis=2))
    corr = cov / std_product

    return corr


class TBiGruApproach(AdaptionApproach):
    def __init__(self, lr: float, mmd_factor: float):
        super().__init__()

        self.lr = lr
        self.mmd_factor = mmd_factor

        # training metrics
        self.train_source_loss = torchmetrics.MeanSquaredError()
        self.mmd_loss = rul_adapt.loss.MaximumMeanDiscrepancyLoss(num_kernels=5)

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

    def configure_optimizers(self) -> torch.optim.Adam:
        return torch.optim.Adam(self.parameters(), self.lr)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Predict the RUL values for a batch of input features."""
        return self.regressor(self.feature_extractor(inputs))

    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
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
