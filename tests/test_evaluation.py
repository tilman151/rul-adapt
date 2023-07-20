import numpy as np
import pandas as pd
import pytest

from rul_adapt.evaluation import friedman_nemenyi


@pytest.fixture()
def dataset_names():
    return [f"Data {x}" for x in "ABCDEFGHIJ"]


@pytest.fixture()
def approach_names():
    return [f"Approach {x}" for x in "XYZ"]


def test_friedman_nemenyi_no_rejection(dataset_names, approach_names):
    performances = pd.DataFrame(
        [np.random.randn(10) for _ in range(3)],
        columns=dataset_names,
        index=approach_names,
    )

    avg_ranks, pairwise_significance = friedman_nemenyi(performances)

    assert avg_ranks is None
    assert pairwise_significance is None


def test_friedman_nemenyi_rejection(dataset_names, approach_names):
    performances = pd.DataFrame(
        [np.random.randn(10) + 10 * i for i in range(3)],
        columns=dataset_names,
        index=approach_names,
    )

    avg_ranks, pairwise_significance = friedman_nemenyi(performances)

    assert avg_ranks.shape == (3,)
    assert avg_ranks.values.tolist() == [1.0, 2.0, 3.0]

    assert pairwise_significance.shape == (3, 3)
    assert np.allclose(pairwise_significance.values.diagonal(), 1.0)
