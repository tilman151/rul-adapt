import warnings
from typing import List

import rul_datasets
import torch


def generate_pseudo_labels(
    dm: rul_datasets.RulDataModule, model: torch.nn.Module
) -> List[float]:
    features, _ = dm.load_split("dev")
    last_timestep = torch.stack([f[-1] for f in features])
    pseudo_labels = model(last_timestep).squeeze(axis=1)

    if dm.reader.max_rul and torch.any(pseudo_labels > dm.reader.max_rul):
        warnings.warn(
            "At least one of the generated pseudo labels is greater "
            "than the maximum RUL of the dataset. This may lead to unexpected results "
            "when trying to instantiate a reader with these pseudo labels."
        )
    if torch.any(pseudo_labels < 0):
        warnings.warn(
            "At least one of the generated pseudo labels is negative. "
            "Please consider clipping them to zero."
        )

    return pseudo_labels.tolist()
