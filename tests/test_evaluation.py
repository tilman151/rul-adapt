import numpy as np
import pandas as pd
import pytest

from rul_adapt.evaluation import friedman_nemenyi, plot_critical_difference


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

    assert avg_ranks is not None
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


def test_critical_difference_plot():
    avg_ranks = pd.Series(
        [1.5, 2.5, 3.5, 3.7], index=["AAAAAAAAAAA", "BBBBBB", "CCCCC", "D"]
    )
    pairwise_significance = pd.DataFrame(
        [
            [1.0, 0.1, 0.01, 0.04],
            [0.1, 1.0, 0.1, 0.04],
            [0.01, 0.1, 1.0, 0.4],
            [0.04, 0.04, 0.4, 1.0],
        ],
        index=["AAAAAAAAAAA", "BBBBBB", "CCCCC", "D"],
        columns=["AAAAAAAAAAA", "BBBBBB", "CCCCC", "D"],
    )

    fig = plot_critical_difference(
        avg_ranks, pairwise_significance, annotation_ratio=0.3, figsize=(7, 2)
    )
    fig.show()
