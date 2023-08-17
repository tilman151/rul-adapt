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