from copy import deepcopy
from typing import Tuple, Any, Dict

import rul_datasets
import pytorch_lightning as pl

import rul_adapt
from rul_adapt.approach import DannApproach

BATCH_SIZE = {
    (1, 2): 256,
    (1, 3): 256,
    (1, 4): 256,
    (2, 1): 512,
    (2, 3): 256,
    (2, 4): 256,
    (3, 1): 256,
    (3, 2): 256,
    (3, 4): 256,
    (4, 1): 512,
    (4, 2): 512,
    (4, 3): 512,
}
FEATURE_EXTRACTOR = {
    (1, 2): {
        "lstm_units": [128],
        "fc_units": 64,
        "lstm_dropout": 0.5,
        "fc_dropout": 0.5,
    },
    (1, 3): {
        "lstm_units": [128],
        "fc_units": 64,
        "lstm_dropout": 0.5,
        "fc_dropout": 0.5,
    },
    (1, 4): {
        "lstm_units": [128],
        "fc_units": 64,
        "lstm_dropout": 0.7,
        "fc_dropout": 0.7,
    },
    (2, 1): {
        "lstm_units": [64],
        "fc_units": 64,
        "lstm_dropout": 0.1,
        "fc_dropout": 0.1,
    },
    (2, 3): {
        "lstm_units": [64],
        "fc_units": 512,
        "lstm_dropout": 0.1,
        "fc_dropout": 0.1,
    },
    (2, 4): {
        "lstm_units": [32, 32],
        "fc_units": 32,
        "lstm_dropout": 0.1,
        "fc_dropout": 0.1,
    },
    (3, 1): {
        "lstm_units": [64, 32],
        "fc_units": 128,
        "lstm_dropout": 0.3,
        "fc_dropout": 0.3,
    },
    (3, 2): {
        "lstm_units": [64, 32],
        "fc_units": 64,
        "lstm_dropout": 0.3,
        "fc_dropout": 0.3,
    },
    (3, 4): {
        "lstm_units": [64, 32],
        "fc_units": 64,
        "lstm_dropout": 0.3,
        "fc_dropout": 0.3,
    },
    (4, 1): {
        "lstm_units": [100],
        "fc_units": 30,
        "lstm_dropout": 0.5,
        "fc_dropout": 0.5,
    },
    (4, 2): {
        "lstm_units": [100],
        "fc_units": 30,
        "lstm_dropout": 0.5,
        "fc_dropout": 0.5,
    },
    (4, 3): {
        "lstm_units": [100],
        "fc_units": 30,
        "lstm_dropout": 0.5,
        "fc_dropout": 0.5,
    },
}
REGRESSOR = {
    (1, 2): {"units": [32, 1], "dropout": 0.3},
    (1, 3): {"units": [32, 1], "dropout": 0.3},
    (1, 4): {"units": [32, 32, 1], "dropout": 0.3},
    (2, 1): {"units": [32, 1], "dropout": 0.0},
    (2, 3): {"units": [64, 32, 1], "dropout": 0.0},
    (2, 4): {"units": [32, 1], "dropout": 0.0},
    (3, 1): {"units": [32, 32, 1], "dropout": 0.1},
    (3, 2): {"units": [32, 32, 1], "dropout": 0.1},
    (3, 4): {"units": [32, 32, 1], "dropout": 0.1},
    (4, 1): {"units": [20, 1], "dropout": 0.0},
    (4, 2): {"units": [20, 1], "dropout": 0.0},
    (4, 3): {"units": [20, 1], "dropout": 0.0},
}
DISC = {
    (1, 2): {"units": [32, 1], "dropout": 0.3},
    (1, 3): {"units": [32, 1], "dropout": 0.3},
    (1, 4): {"units": [32, 1], "dropout": 0.3},
    (2, 1): {"units": [16, 16, 1], "dropout": 0.1},
    (2, 3): {"units": [64, 32, 1], "dropout": 0.1},
    (2, 4): {"units": [16, 1], "dropout": 0.1},
    (3, 1): {"units": [32, 32, 1], "dropout": 0.1},
    (3, 2): {"units": [32, 32, 1], "dropout": 0.1},
    (3, 4): {"units": [32, 32, 1], "dropout": 0.1},
    (4, 1): {"units": [20, 1], "dropout": 0.1},
    (4, 2): {"units": [20, 1], "dropout": 0.1},
    (4, 3): {"units": [20, 1], "dropout": 0.1},
}
DANN = {
    (1, 2): {"dann_factor": 0.8, "lr": 0.01, "weight_decay": 0.01},
    (1, 3): {"dann_factor": 0.8, "lr": 0.01, "weight_decay": 0.01},
    (1, 4): {"dann_factor": 1.0, "lr": 0.01, "weight_decay": 0.01},
    (2, 1): {"dann_factor": 1.0, "lr": 0.01, "weight_decay": 0.01},
    (2, 3): {"dann_factor": 2.0, "lr": 0.1, "weight_decay": 0.01},
    (2, 4): {"dann_factor": 1.0, "lr": 0.1, "weight_decay": 0.01},
    (3, 1): {"dann_factor": 2.0, "lr": 0.01, "weight_decay": 0.01},
    (3, 2): {"dann_factor": 2.0, "lr": 0.01, "weight_decay": 0.01},
    (3, 4): {"dann_factor": 2.0, "lr": 0.01, "weight_decay": 0.01},
    (4, 1): {"dann_factor": 1.0, "lr": 0.01, "weight_decay": 0.01},
    (4, 2): {"dann_factor": 1.0, "lr": 0.01, "weight_decay": 0.01},
    (4, 3): {"dann_factor": 1.0, "lr": 0.01, "weight_decay": 0.01},
}


def get_lstm_dann(
    source_fd: int, target_fd: int, **trainer_kwargs: Any
) -> Tuple[rul_datasets.DomainAdaptionDataModule, DannApproach, pl.Trainer]:
    """
    Construct an LSTM-DANN approach for CMAPSS with the original hyperparameters.

    Examples:
        ```pycon
        >>> import rul_adapt
        >>> dm, dann, trainer = rul_adapt.construct.get_lstm_dann(3, 1)
        >>> trainer.fit(dann, dm)
        >>> trainer.test(dann, dm)
        ```

    Args:
        source_fd: The source FD of CMAPSS.
        target_fd: The target FD of CMAPSS.
        trainer_kwargs: Overrides for the trainer class.
    Returns:
        dm: The data module for adaption of two CMAPSS sub-datasets.
        dann: The DANN approach with feature extractor, regressor and domain disc.
        trainer: The trainer object.
    """
    config = get_lstm_dann_config(source_fd, target_fd)
    config["trainer"].update(trainer_kwargs)
    dm, dann, trainer = lstm_dann_from_config(config)

    return dm, dann, trainer


def get_lstm_dann_config(source_fd: int, target_fd: int) -> Dict[str, Any]:
    """
    Get a configuration for the LSTM-DANN approach.

    The configuration can be modified and fed to [lstm_dann_from_config]
    [rul_adapt.construct.lstm_dann.lstm_dann_from_config] to create the approach.

    Args:
        source_fd: The source FD of CMAPSS.
        target_fd: The target FD of CMAPSS.
    Returns:
        The LSTM-DANN configuration.
    """
    _validate(source_fd, target_fd)

    source_config = {"fd": source_fd, "feature_select": list(range(24))}
    target_config = {"fd": target_fd}
    dm_config = {
        "source": source_config,
        "target": target_config,
        "batch_size": BATCH_SIZE[(source_fd, target_fd)],
    }

    fe_config = FEATURE_EXTRACTOR[(source_fd, target_fd)]
    fe_config["input_channels"] = 24

    reg_config = REGRESSOR[(source_fd, target_fd)]
    reg_config["input_channels"] = fe_config["fc_units"]
    reg_config["act_func_on_last_layer"] = False

    disc_config = DISC[(source_fd, target_fd)]
    disc_config["input_channels"] = fe_config["fc_units"]
    disc_config["act_func_on_last_layer"] = False

    dann_config = DANN[(source_fd, target_fd)]
    dann_config["lr_decay_factor"] = 0.1
    dann_config["lr_decay_epochs"] = 100

    trainer_config = {
        "max_epochs": 200,
        "gradient_clip_val": 1.0,
        "callbacks": [pl.callbacks.EarlyStopping("val/target_rmse", patience=20)],
    }

    config = {
        "dm": dm_config,
        "dann": dann_config,
        "domain_disc": disc_config,
        "feature_extractor": fe_config,
        "regressor": reg_config,
        "trainer": trainer_config,
    }

    return deepcopy(config)


def lstm_dann_from_config(
    config: Dict[str, Any]
) -> Tuple[rul_datasets.DomainAdaptionDataModule, DannApproach, pl.Trainer]:
    """
    Construct a LSTM-DANN approach from a configuration.

    The configuration can be created by calling [get_lstm_dann_config]
    [rul_adapt.construct.lstm_dann.get_lstm_dann_config].

    Args:
        config: The LSTM-DANN configuration.
    Returns:
        dm: The data module for adaption of two CMAPSS sub-datasets.
        dann: The DANN approach with feature extractor, regressor and domain disc.
        trainer: The trainer object.
    """
    source = rul_datasets.CmapssReader(**config["dm"]["source"])
    target = source.get_compatible(**config["dm"]["target"])
    dm = rul_datasets.DomainAdaptionDataModule(
        rul_datasets.RulDataModule(source, config["dm"]["batch_size"]),
        rul_datasets.RulDataModule(target, config["dm"]["batch_size"]),
    )

    feature_extractor = rul_adapt.model.LstmExtractor(**config["feature_extractor"])
    regressor = rul_adapt.model.FullyConnectedHead(**config["regressor"])
    domain_disc = rul_adapt.model.FullyConnectedHead(**config["domain_disc"])

    dann = rul_adapt.approach.DannApproach(**config["dann"])
    dann.set_model(feature_extractor, regressor, domain_disc)

    trainer = pl.Trainer(**config["trainer"])

    return dm, dann, trainer


def _validate(source_fd: int, target_fd: int) -> None:
    if source_fd == target_fd:
        raise ValueError(
            f"No configuration for adapting from FD{source_fd:03} to itself."
        )
    elif 1 > source_fd or source_fd > 4:
        raise ValueError(f"CMAPSS has only FD001 to FD004 but no FD{source_fd:03}")
    elif 1 > target_fd or target_fd > 4:
        raise ValueError(f"CMAPSS has only FD001 to FD004 but no FD{target_fd:03}")
