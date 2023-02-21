"""This module contains the implementation of Maximal Overlap Discrete Wavelet 
Transform (MODWT)."""

import numpy as np
import pywt


def modwt(inputs: np.ndarray, wavelet: str, level: int) -> np.ndarray:
    """
    Apply Maximal Overlap Discrete Wavelet Transformation (MODWT) of `level` to the
    input.

    The `wavelet` should be a string that can be passed to `pywt` to construct a
    wavelet function. For more options call `pywt.wavelist`.

    Args:
        inputs: An input signal of shape `[num_windows, window_size, num_features]`.
        wavelet: The description of the wavelet function, e.g. 'sym4'.
        level: The decomposition level.

    Returns:
        The decompositions in the last axis ordered as D1, A1, D2, A2, ..., Dn, An.
    """
    wavelet = pywt.Wavelet(wavelet)
    dec_hi = np.array(wavelet.dec_hi) / np.sqrt(2)
    dec_lo = np.array(wavelet.dec_lo) / np.sqrt(2)
    coeffs = []
    approx = inputs
    for j in range(level):
        detail = _circular_convolve_fast(dec_hi, approx, j + 1)
        approx = _circular_convolve_fast(dec_lo, approx, j + 1)
        coeffs.append(detail)
        coeffs.append(approx)

    return np.concatenate(coeffs, axis=-1)


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
