import numpy as np
import pywt


def modwt(x, filters, level):
    """
    filters: 'db1', 'db2', 'haar', ...
    return: see matlab
    """
    # filter
    wavelet = pywt.Wavelet(filters)
    dec_hi = np.array(wavelet.dec_hi) / np.sqrt(2)
    dec_lo = np.array(wavelet.dec_lo) / np.sqrt(2)
    wavecoeff = []
    v_j_1 = x
    for j in range(level):
        w = circular_convolve_fast(dec_hi, v_j_1, j + 1)
        v_j_1 = circular_convolve_fast(dec_lo, v_j_1, j + 1)
        wavecoeff.append(w)
        wavecoeff.append(v_j_1)

    return np.concatenate(wavecoeff, axis=-1)


def circular_convolve_d(h_t, v_j_1, j):
    """
    jth level decomposition
    h_t: \tilde{h} = h / sqrt(2)
    v_j_1: v_{j-1}, the (j-1)th scale coefficients
    return: w_j (or v_j)
    """
    len_signal = len(v_j_1)
    len_wavelet = len(h_t)
    w_j = np.zeros(len_signal)
    l = np.arange(len_wavelet)
    for t in range(len_signal):
        index = np.mod(t - 2 ** (j - 1) * l, len_signal)
        v_p = np.array([v_j_1[ind] for ind in index])
        w_j[t] = (np.array(h_t) * v_p).sum()

    return w_j


def circular_convolve_fast(kernel, signal, level):
    len_signal = signal.shape[-2]
    signal_range = np.arange(len_signal)[:, None]
    wavelet_range = np.arange(len(kernel))[None, :]
    idx = np.mod(signal_range - 2 ** (level - 1) * wavelet_range, len_signal)
    w_j = np.sum(kernel[None, :, None] * signal[..., idx, :], axis=-2)

    return w_j
