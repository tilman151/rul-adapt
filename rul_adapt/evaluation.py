import json
import os.path
from typing import Tuple, Optional, Dict, Any

import networkx as nx
import pandas as pd
import scipy
import scikit_posthocs as sp
import wandb
import tempfile

import matplotlib.pyplot as plt


def get_best_tune_run(run_path: str) -> Dict[str, Any]:
    """
    Get the best trial from a tune run, according to Friedman-Nemenyi.

    Args:
        run_path: path to the wandb tune summary run
    Returns:
        best_trial: dict with the best trial's config
    """
    table = load_trials_table(run_path)
    performances = trials2performances(table)
    avg_ranks, significances = friedman_nemenyi(performances)
    best_trial_id = avg_ranks.idxmin()
    best_trial = table.loc[best_trial_id][
        [c for c in table.columns if c.startswith("config/")]
    ].to_dict()

    return best_trial


def load_trials_table(run_path: str) -> pd.DataFrame:
    """
    Load the trial table from a wandb tune summary run.
    Args:
        run_path: path to the wandb tune summary run

    Returns:
        table: DataFrame with the trials' performance and config values
    """
    run = wandb.Api().run(run_path)
    with tempfile.TemporaryDirectory() as tmpdir:
        table_dir = run.logged_artifacts()[0].download(root=tmpdir)
        with open(os.path.join(table_dir, "tune_analysis.table.json")) as f:
            table_json = json.load(f)
    table = pd.DataFrame(table_json["data"], columns=table_json["columns"])
    table = table.set_index("trial_id")

    return table


def trials2performances(table: pd.DataFrame) -> pd.DataFrame:
    """Extract the performance values from a trial table."""
    return table[[c for c in table.columns if c.startswith("rmse")]]


def friedman_nemenyi(
    performance: pd.DataFrame, p: float = 0.05
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Friedman-Nemenyi test for multiple comparison of algorithms.

    Args:
        performance: DataFrame with performance values of algorithms
        p: significance level for Friedman test
    Returns:
        avg_ranks: average ranking of algorithms
        pairwise_significance: p-values of pairwise comparisons
    """
    _, friedman_pvalue = scipy.stats.friedmanchisquare(*performance.values)

    approaches = performance.index.tolist()
    datasets = performance.columns.tolist()
    ranks = pd.DataFrame(
        scipy.stats.rankdata(performance, axis=0), columns=datasets, index=approaches
    )
    avg_ranks = ranks.mean(axis=1)
    if friedman_pvalue > p:
        print("Friedman test: No significant difference between approaches.")
        pairwise_significance = sp.posthoc_nemenyi_friedman(performance.T)
    else:
        pairwise_significance = None

    return avg_ranks, pairwise_significance


def plot_critical_difference(
    avg_ranks: pd.Series,
    pairwise_significance: pd.DataFrame,
    annotation_ratio: float = 0.5,
    **kwargs
) -> plt.Figure:
    fig: plt.Figure = plt.figure(**kwargs)

    ax: plt.Axes = fig.gca()
    _annotate_ranks(ax, avg_ranks, annotation_ratio)
    _connect_cliques(ax, avg_ranks, pairwise_significance, annotation_ratio)

    # right-to-left x-axis on top
    ax.set_xticks(range(1, len(avg_ranks) + 2))
    ax.xaxis.tick_top()
    ax.invert_xaxis()

    # invisible y-axis
    ax.set_ylim(0.0, 1.0)
    ax.invert_yaxis()
    ax.get_yaxis().set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # fit to content
    fig.tight_layout()

    return fig


def _annotate_ranks(ax, avg_ranks, ratio):
    arrowprops = dict(arrowstyle="-", connectionstyle="angle,angleA=0,angleB=90")
    min_x, max_x = 1, len(avg_ranks) + 1
    for i, (approach, rank) in enumerate(avg_ranks.sort_values().items()):
        text_pos = _get_text_pos(i, len(avg_ranks), min_x, max_x, ratio)
        ax.annotate(
            approach,
            xy=(rank, 0.0),
            xytext=text_pos,
            ha="left" if text_pos[0] == min_x else "right",
            va="bottom",
            arrowprops=arrowprops,
        )


def _connect_cliques(ax, avg_ranks, pairwise_significance, annotation_ratio):
    min_text_pos = _get_min_text_pos(len(avg_ranks), annotation_ratio)
    pairwise_significance = pairwise_significance > 0.05
    cliques = nx.find_cliques(nx.to_networkx_graph(pairwise_significance))
    cliques = [
        (avg_ranks[c].max() + 0.05, avg_ranks[c].min() - 0.05)
        for c in cliques
        if len(c) > 1
    ]
    cliques.sort(reverse=True)
    num_cliques = len(cliques)
    offset = 0.1 * min_text_pos
    for i, (max_rank, min_rank) in enumerate(cliques):
        y_pos = offset + 0.8 * min_text_pos * i / num_cliques
        ax.hlines(y_pos, min_rank, max_rank, color="black", linewidth=4)


def _get_text_pos(i, max_items, min_x, max_x, ratio=0.5):
    direction_switch_idx = _get_direction_switch_idx(max_items)
    max_annotations_per_side = _get_max_annotations_per_side(max_items)
    if i < direction_switch_idx:
        x_pos = min_x
        y_pos_idx = i
    else:
        x_pos = max_x
        y_pos_idx = max_items - (i + 1)
    text_pos = (x_pos, (1 - ratio) + ratio * y_pos_idx / (max_annotations_per_side - 1))

    return text_pos


def _get_max_annotations_per_side(max_items):
    direction_switch_idx = _get_direction_switch_idx(max_items)
    max_annotations = direction_switch_idx + max_items % 2

    return max_annotations


def _get_direction_switch_idx(max_items):
    return max_items // 2


def _get_min_text_pos(max_items, annotation_ratio):
    return _get_text_pos(max_items - 1, max_items, 0, 1, annotation_ratio)[1]
