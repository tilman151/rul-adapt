"""The TBiGRU approach uses a feature selection mechanism to mine transferable
features and a bearing running state detection to determine the
first-time-to-predict. The training is done with an [MMD approach]
[rul_adapt.approach.mmd].

The feature selection uses a distance measure based on Dynamic Time Warping and the
Wasserstein distance. From a set of 30 common vibration features the ones with the
smallest distance between source and target domain are selected. These features serve
as inputs to the network.

The first-time-to-predict (FTTP) is used to generate the RUL labels for training.
FTTP is the time step where the degradation can be detected for the first time. The
RUL labels before this time step should be constant. The TBiGRU approach uses the
moving average correlation (MAC) of the energy entropies of four levels of maximal
overlap discrete wavelet transform (MODWT) decompositions to determine four running
states of each bearing. The end of the steady running state marks the FTTP.

TBiGRU was introduced by [Cao et al.](
https://doi.org/10.1016/j.measurement.2021.109287) and evaluated on the FEMTO Bearing
dataset."""

from itertools import product
from queue import Queue
from typing import List, Tuple, Optional

import numpy as np
import pywt  # type: ignore
import scipy.stats  # type: ignore
from dtaidistance import dtw  # type: ignore
from rul_datasets.reader import AbstractReader
from rul_datasets.utils import extract_windows
from scipy.stats import wasserstein_distance  # type: ignore
from sklearn.preprocessing import MinMaxScaler  # type: ignore


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
    return scipy.stats.kurtosis(inputs, axis=-2, fisher=False)


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
    wp = pywt.WaveletPacket(inputs, wavelet="dmey", maxlevel=4, axis=-2)
    bands = (node.data for node in wp.get_level(4))
    energies = np.concatenate([energy(b) for b in bands], axis=-1)

    return energies


def std_ihc(inputs: np.ndarray) -> np.ndarray:
    return np.std((np.arccosh(inputs + 0j)), axis=-2)


def std_ihs(inputs: np.ndarray) -> np.ndarray:
    return np.std(np.arcsinh(inputs), axis=-2)


class VibrationFeatureExtractor:
    """This class extracts 30 different features from a raw acceleration signal.

    The features are: RMS, kurtosis, peak2peak, standard deviation, skewness,
    margin factor, impulse factor, energy, median absolute, gini factor, maximum
    absolute, mean absolute, energies of the 16 bands resulting from wavelet packet
    decomposition, standard deviation of arccosh and arcsinh. If the input has n
    features, n*30 features are extracted. Additionally, it features a scaler that
    can be fit to scale all extracted features between [0, 1]."""

    _scaler: Optional[MinMaxScaler]

    def __init__(
        self, num_input_features: int, feature_idx: Optional[List[int]] = None
    ) -> None:
        """
        Create a new vibration feature extractor with the selected features.

        The features are sorted as f1_1, .., f1_j, ..., fi_j, where i is the index of
        the computed feature (between 0 and 30) and j is the index of the raw
        feature (between 0 and `num_input_features`).

        Args:
            num_input_features: The number of input features.
            feature_idx: The indices of the features to compute.
        """
        self.num_input_features = num_input_features
        self.feature_idx = list(range(60)) if feature_idx is None else feature_idx

        if min(self.feature_idx) < 0 or max(self.feature_idx) > 60:
            raise ValueError("Feature indices need to be between 0 and 60.")

        self._scaler = None

    def __call__(
        self, features: np.ndarray, targets: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract the features from the input and optionally scale them.

        The features should have the shape `[num_windows, window_size,
        num_input_features]` and the targets `[num_windows]`.

        Args:
            features: The input features.
            targets: The input targets.

        Returns:
            The extracted features and input targets.
        """
        features = self._extract_selected(features)
        if self._scaler is not None:
            features = self._scaler.transform(features)

        return features, targets

    def _extract_selected(self, features: np.ndarray) -> np.ndarray:
        return _extract_all(features, self.num_input_features)[:, self.feature_idx]

    def fit(self, features: List[np.ndarray]) -> "VibrationFeatureExtractor":
        """
        Fit the internal scaler on a list of raw feature time series.

        The time series are passed through the feature extractor and then used to fit
        the internal min-max scaler. Each time series in the list should have the
        shape `[num_windows, window_size, num_input_features]`.

        Args:
            features: The list of raw feature time series.

        Returns:
            The feature extractor itself.
        """
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


def select_features(
    source: AbstractReader, target: AbstractReader, num_features: int
) -> List[int]:
    """
    Select the most transferable features between source and target domain.

    30 features are considered: RMS, kurtosis, peak2peak, standard deviation, skewness,
    margin factor, impulse factor, energy, median absolute, gini factor, maximum
    absolute, mean absolute, energies of the 16 bands resulting from wavelet packet
    decomposition, standard deviation of arccosh and arcsinh. If the input has n raw
    features, n*30 features are extracted.

    The `dev` splits of both domains are used to calculate a distance metric based on
    Dynamic Time Warping and the Wasserstein Distance. The indices of the
    `num_feature` features with the lowest distances are returned.

    Args:
        source: The reader of the source domain.
        target: The reader of the target domain.
        num_features: The number of transferable features to return.

    Returns:
        The indices of features ordered by transferability.
    """
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
            distances[i] += _domain_distance(s, f)

    feature_idx = np.argsort(distances)[:num_features].tolist()

    return feature_idx


def _domain_distance(source: np.ndarray, target: np.ndarray, ratio=0.4) -> float:
    dtw_dist = dtw.distance_fast(source, target)
    wasserstein_dist = wasserstein_distance(source, target)
    distance = ratio * dtw_dist + (1 - ratio) * wasserstein_dist

    return distance


def mac(inputs: np.ndarray, window_size: int, wavelet: str = "dmey") -> np.ndarray:
    """
    Calculate the moving average correlation (MAC) of the energy entropies of four
    levels of maximal overlap discrete wavelet transform (MODWT) decompositions.

    The `wavelet` is a wavelet description that can be passed to `pywt`. The default
    wavelet was confirmed by the original authors. For more options call
    `pywt.wavelist`. The input signal should have the shape `[num_windows,
    window_size, num_features]`.

    Args:
        inputs: The input acceleration signal.
        window_size: The window size of the sliding window to calculate the average
                     over.
        wavelet: The description of the wavelet, e.g. 'sym4'.

    Returns:
        The MAC of the input signal which is `window_size - 1` shorter.
    """
    entropies = _energy_entropies(inputs, wavelet)
    entropies = extract_windows(entropies, window_size)
    anchor, queries = entropies[:, -2:-1], entropies[:, :-1]
    corr = _pearson(anchor, queries)
    corr = np.mean(np.abs(corr), axis=1)

    return corr


def _energy_entropies(inputs: np.ndarray, wavelet: str = "sym4") -> np.ndarray:
    coeffs = modwpt(inputs, wavelet, 4)
    energies = energy(coeffs)
    ratios = energies / np.sum(energies, axis=-1, keepdims=True)
    entropy = -np.sum(ratios * np.log(ratios), axis=-1, keepdims=True)
    entropies = ratios / entropy

    return entropies


def _pearson(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    diff_x = x - np.mean(x, axis=2, keepdims=True)
    diff_y = y - np.mean(y, axis=2, keepdims=True)
    cov = np.sum(diff_y * diff_x, axis=2)
    std_product = np.sqrt(np.sum(diff_y**2, axis=2) * np.sum(diff_x**2, axis=2))
    corr = cov / std_product

    return corr


def modwpt(inputs: np.ndarray, wavelet: str, level: int) -> np.ndarray:
    """
    Apply Maximal Overlap Discrete Wavelet Packet Transformation (MODWT) of `level`
    to the input.

    The `wavelet` should be a string that can be passed to `pywt` to construct a
    wavelet function. For more options call `pywt.wavelist`. The implementation was
    inspired by [this repository](https://github.com/pistonly/modwtpy).

    Args:
        inputs: An input signal of shape `[num_windows, window_size, num_features]`.
        wavelet: The description of the wavelet function, e.g. 'sym4'.
        level: The decomposition level.

    Returns:
        The 2**level decompositions stacked in the last axis.
    """
    if level < 1:
        raise ValueError("The level needs to be a positive integer.")
    wavelet_func = pywt.Wavelet(wavelet)
    dec_hi = np.array(wavelet_func.dec_hi) / np.sqrt(2)
    dec_lo = np.array(wavelet_func.dec_lo) / np.sqrt(2)

    input_queue: Queue = Queue(maxsize=2**level)
    input_queue.put(inputs)
    for j in range(level):
        coeffs = _decompose_level(input_queue, dec_hi, dec_lo, j)
        for d in coeffs:
            input_queue.put(d)

    return np.concatenate(coeffs, axis=-1)


def _decompose_level(input_queue, dec_hi, dec_lo, level):
    coeffs = []
    while not input_queue.empty():
        signal = input_queue.get()
        detail = _circular_convolve_fast(dec_hi, signal, level + 1)
        approx = _circular_convolve_fast(dec_lo, signal, level + 1)
        coeffs.append(approx)
        coeffs.append(detail)

    return coeffs


def _circular_convolve_d(
    kernel: np.ndarray, signal: np.ndarray, level: int
) -> np.ndarray:
    len_signal = len(signal)
    len_wavelet = len(kernel)
    convolved = np.zeros(len_signal)
    wavelet_range = np.arange(len_wavelet)
    for t in range(len_signal):
        index = np.mod(t - 2 ** (level - 1) * wavelet_range, len_signal)
        element = np.array([signal[ind] for ind in index])
        convolved[t] = (np.array(kernel) * element).sum()

    return convolved


def _circular_convolve_fast(
    kernel: np.ndarray, signal: np.ndarray, level: int
) -> np.ndarray:
    len_signal = signal.shape[-2]
    signal_range = np.arange(len_signal)[:, None]
    wavelet_range = np.arange(len(kernel))[None, :]
    idx = np.mod(signal_range - 2 ** (level - 1) * wavelet_range, len_signal)
    convolved = np.sum(kernel[None, :, None] * signal[..., idx, :], axis=-2)

    return convolved
