import json
import os.path
from typing import Tuple, Optional, Dict, Any

import pandas as pd
import scipy
import scikit_posthocs as sp
import wandb
import tempfile


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
    return table[[c for c in table.columns if "rmse" in c]]


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
    if friedman_pvalue > p:
        print("Friedman test: No significant difference between approaches.")
        return None, None

    approaches = performance.index.tolist()
    datasets = performance.columns.tolist()
    ranks = pd.DataFrame(
        scipy.stats.rankdata(performance, axis=0), columns=datasets, index=approaches
    )
    avg_ranks = ranks.mean(axis=1)
    pairwise_significance = sp.posthoc_nemenyi_friedman(performance.T)

    return avg_ranks, pairwise_significance
