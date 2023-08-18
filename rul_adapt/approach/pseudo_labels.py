"""Pseudo labeling is a simple approach that takes a model trained on the source
domain to label the target domain. Afterward, the model training is continued on the
combined source and target data. This process is repeated until the validation loss
converges.

Used In:
    * Wang et al. (2022). **Residual Life Prediction of Bearings Based on SENet-TCN
    and Transfer Learning.** *IEEE Access*, *10*,
    [10.1109/ACCESS.2022.3223387](https://doi.org/10.1109/ACCESS.2022.3223387)

Examples:
    ```python
    import torch
    import rul_datasets
    import pytorch_lightning as pl

    from rul_adapt import model
    from rul_adapt import approach

    feat_ex = model.CnnExtractor(14, [16], 30, fc_units=16)
    reg = model.FullyConnectedHead(16, [1])

    supervised = approach.SupervisedApproach(0.001, "rmse", "adam")
    supervised.set_model(feat_ex, reg)

    fd1 = rul_datasets.RulDataModule(rul_datasets.CmapssReader(1), 32)
    fd1.setup()
    fd3 = rul_datasets.RulDataModule(rul_datasets.CmapssReader(3), 32)
    fd3.setup()

    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(supervised, fd3)

    pseudo_labels = approach.generate_pseudo_labels(fd1, supervised)
    pseudo_labels = [max(0, min(125, pl)) for pl in pseudo_labels]
    approach.patch_pseudo_labels(fd3, pseudo_labels)

    combined_data = torch.utils.data.ConcatDataset(
        [fd1.to_dataset("dev"), fd3.to_dataset("dev")]
    )
    combined_dl = torch.utils.data.DataLoader(combined_data, batch_size=32)

    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(supervised, train_dataloader=combined_dl)
    ```
"""

import warnings
from typing import List, Tuple, Optional, Union

import numpy as np
import rul_datasets
import torch


def generate_pseudo_labels(
    dm: rul_datasets.RulDataModule, model: torch.nn.Module, inductive: bool = False
) -> List[float]:
    """
    Generate pseudo labels for the dev set of a data module.

    The pseudo labels are generated for the last timestep of each run. They are
    returned raw and may therefore contain values bigger than `max_rul` or negative
    values. It is recommended to clip them to zero and `max_rul` respectively before
    using them to patch a reader.

    Args:
        dm: The data module to generate pseudo labels for.
        model: The model to use for generating the pseudo labels.
        inductive: Whether to generate pseudo labels for inductive adaption,
                   i.e., use 'test' instead of 'dev' split.

    Returns:
        A list of pseudo labels for the dev set of the data module.
    """
    features, _ = dm.load_split("test" if inductive else "dev", alias="dev")
    last_timestep = torch.stack([f[-1] for f in features])
    pseudo_labels = model(last_timestep).squeeze(axis=1)

    max_rul = get_max_rul(dm.reader)
    if torch.any(pseudo_labels > max_rul):
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


def patch_pseudo_labels(
    dm: rul_datasets.RulDataModule, pseudo_labels: List[float], inductive: bool = False
) -> None:
    """
    Patch a data module with pseudo labels in-place.

    The pseudo labels are used to replace the RUL targets of the dev set of the data
    module. The validation and test sets are not affected.

    It is not possible to patch the same data module multiple times. Instead,
    instantiate a fresh data module and patch that one.

    Args:
        dm: The data module to patch.
        pseudo_labels: The pseudo labels to use for patching the data module.
        inductive: Whether to generate pseudo labels for inductive adaption,
                   i.e., use 'test' instead of 'dev' split.
    """
    if isinstance(dm.reader, _PseudoLabelReader):
        raise RuntimeError(
            "The data module has already been patched with pseudo labels. "
            "Please instantiate a fresh one."
        )
    dm._reader = _PseudoLabelReader(dm.reader, pseudo_labels, inductive)
    dm.setup()  # overwrites any previously loaded data


class _PseudoLabelReader(rul_datasets.reader.AbstractReader):
    """Reader to wrap another reader and replace the RUL targets of the dev set."""

    def __init__(
        self,
        reader: rul_datasets.reader.AbstractReader,
        pseudo_labels: List[float],
        inductive: bool,
    ) -> None:
        super().__init__(
            reader.fd,
            reader.window_size,
            reader.max_rul,
            reader.percent_broken,
            reader.percent_fail_runs,
            reader.truncate_val,
            reader.truncate_degraded_only,
        )

        if any(pl < 0 for pl in pseudo_labels):
            raise ValueError("Pseudo labels must be non-negative.")
        max_rul = get_max_rul(reader)
        if any(pl > max_rul for pl in pseudo_labels):
            raise ValueError(
                "Pseudo labels must be smaller than the maximum RUL "
                f"of the dataset, {max_rul}."
            )

        self._reader = reader
        self._pseudo_labels = pseudo_labels
        self._inductive = inductive
        self._max_rul = max_rul

    @property
    def _patched_split(self) -> str:
        return "test" if self._inductive else "dev"

    @property
    def first_time_to_predict(self) -> Optional[List[int]]:
        if hasattr(self._reader, "first_time_to_predict"):
            return self._reader.first_time_to_predict
        else:
            return None

    @property
    def _dev_fttp(self) -> Optional[List[int]]:
        if self.first_time_to_predict is None:
            return None
        else:
            split = self._patched_split
            return [
                self.first_time_to_predict[i - 1]
                for i in self._reader._preparator.run_split_dist[split]  # type: ignore
            ]

    @property
    def norm_rul(self) -> bool:
        if hasattr(self._reader, "norm_rul"):
            return self._reader.norm_rul
        else:
            return False

    @property
    def fds(self) -> List[int]:
        return self._reader.fds

    def prepare_data(self) -> None:
        self._reader.prepare_data()

    def default_window_size(self, fd: int) -> int:
        return self._reader.default_window_size(fd)

    def load_split(
        self, split: str, alias: Optional[str] = None
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        alias = alias or split
        features, targets = self._reader.load_split(split, alias)
        if split == self._patched_split and alias == "dev":
            targets = self._get_pseudo_labels(features)

        return features, targets

    def load_complete_split(
        self, split: str, alias: str
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        return self._reader.load_complete_split(split, alias)

    def _get_pseudo_labels(self, features: List[np.ndarray]) -> List[np.ndarray]:
        if not len(features) == len(self._pseudo_labels):
            raise RuntimeError(
                f"Number of runs ({len(features)}) and pseudo labels "
                f"({len(self._pseudo_labels)}) do not match."
            )

        if self._dev_fttp is None:
            targets = [
                self._expand_pseudo_label(f, pl)
                for f, pl in zip(features, self._pseudo_labels)
            ]
        else:
            targets = [
                self._expand_pseudo_label_fttp(f, pl, fttp)
                for f, pl, fttp in zip(features, self._pseudo_labels, self._dev_fttp)
            ]

        return targets

    def _expand_pseudo_label(
        self, feature: np.ndarray, pseudo_label: float
    ) -> np.ndarray:
        first_rul = pseudo_label + len(feature)
        rul_values = np.arange(pseudo_label, first_rul)[::-1]
        if self.max_rul is not None:
            rul_values = np.minimum(self.max_rul, rul_values)

        return rul_values

    def _expand_pseudo_label_fttp(
        self, feature: np.ndarray, pseudo_label: float, fttp: int
    ) -> np.ndarray:
        if self.norm_rul:
            fttp -= 1  # linspace includes 1.0 so decrease starts earlier
            rul_values = np.empty(len(feature))
            rul_values[:fttp] = 1.0
            remaining_steps = max(0, len(feature) - fttp)
            rul_values[fttp:] = np.linspace(1.0, pseudo_label, remaining_steps)
        else:
            first_rul = pseudo_label + len(feature)
            max_rul = first_rul - fttp
            rul_values = np.arange(pseudo_label, first_rul)[::-1]
            rul_values = np.minimum(rul_values, max_rul)

        return rul_values


def get_max_rul(reader: rul_datasets.reader.AbstractReader) -> Union[int, float]:
    """Resolve the maximum RUL of a reader to be comparable to floats."""
    return reader.max_rul or (
        1 if hasattr(reader, "norm_rul") and reader.norm_rul else float("inf")
    )
