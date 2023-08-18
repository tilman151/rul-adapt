# RUL Adapt

[![Master](https://github.com/tilman151/rul-adapt/actions/workflows/on_push.yaml/badge.svg)](https://github.com/tilman151/rul-adapt/actions/workflows/on_push.yaml)
[![Release](https://github.com/tilman151/rul-adapt/actions/workflows/on_release.yaml/badge.svg)](https://github.com/tilman151/rul-adapt/actions/workflows/on_release.yaml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This library contains a collection of unsupervised domain adaption algorithms for RUL estimation.
They are provided as [LightningModules](https://pytorch-lightning.readthedocs.io/en/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule) to be used in [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/).

Currently, five approaches are implemented, including their original hyperparameters:

* **LSTM-DANN** by Da Costa et al. (2020)
* **ADARUL** by Ragab et al. (2020)
* **LatentAlign** by Zhang et al. (2021)
* **TBiGRU** by Cao et al. (2021)
* **Consistency-DANN** by Siahpour et al. (2022)

Three approaches are implemented without their original hyperparameters:

* **ConditionalDANN** by Cheng et al. (2021)
* **ConditionalMMD** by Cheng et al. (2021)
* **PseudoLabels** as used by Wang et al. (2022)

This includes the following general approaches adapted for RUL estimation:

* **Domain Adaption Neural Networks (DANN)** by Ganin et al. (2016)
* **Multi-Kernel Maximum Mean Discrepancy (MMD)** by Long et al. (2015)

Each approach has an example notebook which can be found in the [examples](https://github.com/tilman151/rul-adapt/tree/master/examples) folder.

## Installation

This library is pip-installable. Simply type:

```bash
pip install rul-adapt
```

## Contribution

Contributions are always welcome. Whether you want to fix a bug, add a feature or a new approach, just open an issue and a PR.

## Development

This project is set up with [Poetry](https://python-poetry.org/).
It is the easiest to install Poetry via pipx:

```bash
pipx install poetry
```

To install the dependencies, run:

```bash
poetry install
```

This will generate a new virtual environment for you to use.
To activate it, run:

```bash
poetry shell
```

or prefix your commands with `poetry run`.

To run a hyperparameter search for a specific approach on a GPU, run:

```bash
poetry run python tune_adaption.py --dataset <Dataset> --backbone <Backbone> --approach <Approach> --gpu --sweep_name <Name_for_your_sweep> --entity <WandB_Entity>
```

All results will be logged to WandB in the specified entity and project.
By default, CMAPSS runs with four parallel trials and the remaining datasets with one.
To change this, go to line 53 or 56 respectively and set the value for `"gpu"`.
If you want five parallel trials, set it to 0.2.
How many trials can be run in parallel depends on the GPU memory.

Each trial will be logged, and after all of them are finished, an additional summary run will be created.
This run contains the analysis dataframe of the search.
To get the best hyperparameters, you can run:

```python
from rul_adapt.evaluation import get_best_tune_run

best_hparams = get_best_tune_run("<WandB_Summary_Run_Path>")
```

The returned dictionary contains the best hyperparameters.