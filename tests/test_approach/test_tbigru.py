import numpy as np
import numpy.testing as npt
import pytest
import pywt
import rul_datasets
import pytorch_lightning as pl
import scipy.stats
import torch
from rul_datasets import utils

import rul_adapt.model
from rul_adapt.approach import tbigru, modwt


@pytest.fixture(params=[(4096, 2), (10, 4096, 2)], ids=["unbatched", "batched"])
def inputs_normal(request):
    return np.random.default_rng(seed=42).standard_normal(request.param)


@pytest.fixture(params=[(5000, 2), (10, 5000, 2)], ids=["unbatched", "batched"])
def inputs_uniform(request):
    return np.random.default_rng(seed=42).uniform(-1, 1, request.param)


def test_rms(inputs_normal):
    rms = tbigru.rms(inputs_normal)
    _assert_shape(rms)


def test_p2p(inputs_uniform):
    p2p = tbigru.p2p(inputs_uniform)
    _assert_shape(p2p)
    npt.assert_almost_equal(p2p, 2, decimal=2)


def test_skewness(inputs_normal):
    skew = tbigru.skewness(inputs_normal)
    _assert_shape(skew)
    npt.assert_almost_equal(skew, 0, decimal=1)  # value for standard normal


def test_kurtosis(inputs_normal):
    kurt = tbigru.kurtosis(inputs_normal)
    _assert_shape(kurt)
    npt.assert_almost_equal(kurt, 3, decimal=0)  # value for standard normal


def test_impulse_factor(inputs_uniform):
    imp_f = tbigru.impulse_factor(inputs_uniform)
    _assert_shape(imp_f)
    npt.assert_almost_equal(imp_f, 2, decimal=1)  # max abs of 1 over mean abs of 0.5


def test_median_absolute(inputs_uniform):
    median_abs = tbigru.median_absolute(inputs_uniform)
    _assert_shape(median_abs)
    npt.assert_almost_equal(median_abs, 0.5, decimal=1)


def test_mean_absolute(inputs_uniform):
    mean_abs = tbigru.mean_absolute(inputs_uniform)
    _assert_shape(mean_abs)
    npt.assert_almost_equal(mean_abs, 0.5, decimal=1)


def test_max_absolute(inputs_uniform):
    max_abs = tbigru.max_absolute(inputs_uniform)
    _assert_shape(max_abs)
    npt.assert_almost_equal(max_abs, 1, decimal=1)


def test_std(inputs_normal):
    std = tbigru.std(inputs_normal)
    _assert_shape(std)
    npt.assert_almost_equal(std, 1, decimal=1)


def test_margin_factor(inputs_uniform):
    mf = tbigru.margin_factor(inputs_uniform)
    _assert_shape(mf)
    npt.assert_almost_equal(mf, 2.3, decimal=1)


def test_energy(inputs_normal):
    energy = tbigru.energy(inputs_normal)
    _assert_shape(energy)


def test_gini_factor(inputs_uniform):
    gini = tbigru.gini_factor(inputs_uniform)
    _assert_shape(gini)


@pytest.mark.parametrize(
    "inputs",
    [
        np.random.default_rng(seed=42).random((10, 250, 2)),
        np.random.default_rng(seed=42).uniform(-1, 1, (10, 250, 2)),
    ],
)
def test_gini_factor_approximation(inputs):
    gini = tbigru.gini_factor(inputs)
    accurate_gini = _accurate_gini_factor(inputs)
    npt.assert_almost_equal(gini, accurate_gini)


def _accurate_gini_factor(inputs: np.ndarray) -> np.ndarray:
    """Infeasible due to O(n**2) memory requirements."""
    abs_mean_diff = np.mean(
        np.abs(inputs[:, :, None, :] - inputs[:, None, :, :]), axis=(1, 2)
    )
    rel_mean_diff = abs_mean_diff / np.mean(inputs, axis=1)
    gini = rel_mean_diff / 2

    return gini


def test_band_energies(inputs_normal):
    energies = tbigru.band_energies(inputs_normal)
    _assert_shape(energies, multiplier=16)


def test_std_ihc(inputs_normal):
    ihc = tbigru.std_ihc(inputs_normal)
    _assert_shape(ihc)
    assert np.all(np.isfinite(ihc))
    assert np.all(np.isreal(ihc))


def test_std_ihs(inputs_normal):
    ihs = tbigru.std_ihs(inputs_normal)
    _assert_shape(ihs)
    assert np.all(np.isfinite(ihs))


def _assert_shape(outputs, multiplier=1):
    batched = len(outputs.shape) == 2
    expected_shape = (10, multiplier * 2) if batched else (multiplier * 2,)
    assert outputs.shape == expected_shape


@pytest.mark.parametrize("feature_idx", [[0, 1, 59], None])
def test_feature_extractor(feature_idx):
    inputs = np.random.randn(10, 2560, 2)
    targets = np.arange(10)
    extractor = tbigru.VibrationFeatureExtractor(2, feature_idx)

    outputs, _ = extractor(inputs, targets)

    num_features = len(feature_idx or range(60))
    assert outputs.shape == (10, num_features)


def test_feature_extractor_selection():
    inputs = np.random.randn(10, 2560, 2)
    targets = np.arange(10)
    extractor = tbigru.VibrationFeatureExtractor(2, [1, 25, 59])

    outputs, _ = extractor(inputs, targets)

    npt.assert_almost_equal(outputs[:, 0], tbigru.rms(inputs)[:, 1])
    npt.assert_almost_equal(outputs[:, 1], tbigru.band_energies(inputs)[:, 1])
    npt.assert_almost_equal(outputs[:, 2], tbigru.std_ihs(inputs)[:, 1])


def test_feature_extractor_as_scaler():
    inputs = np.random.default_rng(seed=42).standard_normal((10, 2560, 2))
    targets = np.arange(10)
    extractor = tbigru.VibrationFeatureExtractor(2, [0, 1, 20, 40])
    extractor.fit([inputs])

    outputs, _ = extractor(inputs, targets)

    assert np.max(outputs) <= 1.0
    assert np.min(outputs) >= 0.0


def test_domain_distance():
    source = np.sin(np.linspace(0, 10, 2560))
    target_close = np.sin(2 * np.linspace(0, 10, 2560)) + np.random.randn(2560) * 1e-4
    target_far = np.sin(3 * np.linspace(0, 10, 2560)) + np.random.randn(2560) * 1e-2

    distance = tbigru.domain_distance(source, source)
    assert distance == 0.0

    distance_close = tbigru.domain_distance(source, target_close)
    distance_far = tbigru.domain_distance(source, target_far)
    assert distance_close < distance_far


def test_feature_selection():
    source = rul_datasets.reader.DummyReader(1)
    target = rul_datasets.reader.DummyReader(2)

    feature_idx = tbigru.select_features(source, target, num_features=15)

    assert len(feature_idx) == 15
    assert max(feature_idx) < 30
    assert min(feature_idx) >= 0


def test_circular_convolve_fast():
    h = pywt.Wavelet("db1").dec_hi / np.sqrt(2)
    inputs = np.random.randn(1024)
    for i in range(4):
        fast_outputs = modwt.circular_convolve_fast(h, inputs[:, None], i + 1)
        outputs = modwt.circular_convolve_d(h, inputs, i + 1)
        npt.assert_almost_equal(fast_outputs.squeeze(), outputs)
        inputs = outputs


def test_cicular_convolve_multidim():
    h = pywt.Wavelet("db1").dec_hi / np.sqrt(2)
    inputs = np.random.randn(10, 1024, 3)
    outputs_fast = modwt.circular_convolve_fast(h, inputs, 1)
    outputs = np.empty_like(inputs)
    for w, window in enumerate(inputs):
        for f, feature in enumerate(window.T):
            outputs[w, :, f] = modwt.circular_convolve_d(h, feature, 1)

    npt.assert_almost_equal(outputs_fast, outputs)


def test_energy_entropies(inputs_normal):
    entropies = tbigru.energy_entropies(inputs_normal)
    _assert_shape(entropies, multiplier=8)


def test_pearson():
    x = np.random.randn(1, 1, 16)
    y = np.random.randn(1, 1, 16) + 1
    pearson = tbigru.pearson(x, y)
    expected = scipy.stats.pearsonr(x.squeeze(), y.squeeze())
    assert pearson.shape == (1, 1)
    npt.assert_almost_equal(pearson.squeeze(), expected.statistic)


def test_mac():
    inputs = np.random.default_rng(seed=42).standard_normal((10, 128, 2))
    mac = tbigru.mac(inputs, 9)
    assert mac.shape == (2,)


@pytest.mark.integration
def test_on_dummy():
    torch.autograd.set_detect_anomaly(True)
    source = rul_datasets.reader.DummyReader(1)
    target = source.get_compatible(2, percent_broken=0.8)
    feature_idx = tbigru.select_features(source, target, num_features=15)
    source_extractor = tbigru.VibrationFeatureExtractor(1, feature_idx)
    source_extractor.fit(source.load_split("dev")[0] + target.load_split("dev")[0])
    dm = rul_datasets.DomainAdaptionDataModule(
        rul_datasets.RulDataModule(source, 32, source_extractor, 20),
        rul_datasets.RulDataModule(target, 32, source_extractor, 20),
    )

    fe = rul_adapt.model.CnnExtractor(15, [16, 16], 20, fc_units=16)
    reg = rul_adapt.model.FullyConnectedHead(16, [1], act_func_on_last_layer=False)
    approach = tbigru.TBiGruApproach(0.001, 0.1)
    approach.set_model(fe, reg)

    # workaround because test split has only one window by default
    def _truncate_test_split(rng, features, targets):
        for i in range(len(features)):
            run_len = len(features[i])
            cutoff = rng.integers(run_len // 2, run_len - 1)
            features[i] = features[i][:cutoff]
            targets[i] = targets[i][:cutoff]

        return features, targets

    source._truncate_test_split = _truncate_test_split
    target._truncate_test_split = _truncate_test_split

    trainer = pl.Trainer(
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        max_epochs=10,
    )
    trainer.fit(approach, dm)
    trainer.test(approach, dm)
