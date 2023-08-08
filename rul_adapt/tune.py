import random

from ray import tune


def config2overrides(d, parent_key=""):
    flat = []
    for key, value in d.items():
        key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, dict):
            flat.extend(config2overrides(value, key))
        elif isinstance(value, list):
            flat.append(f"{key}=[{','.join((str(v) for v in value))}]")
        else:
            flat.append(f"{key}={value}")

    return flat


def _get_disc_units():
    return random.randint(0, 2) * ["${regressor.input_channels}"] + [1]


_common = {"training": {"lr": tune.qloguniform(1e-5, 1e-2, 1e-5)}}

dann = {
    "training": {
        "approach": {
            "lr": tune.qloguniform(1e-5, 1e-2, 1e-5),
            "dann_factor": tune.qloguniform(0.1, 10.0, 0.1),
        }
    },
    "domain_disc": {"units": tune.sample_from(_get_disc_units)},
}

adarul = {
    "training": {
        "approach": {
            "lr": tune.qloguniform(1e-5, 1e-2, 1e-5),
            "num_disc_updates": tune.qrandint(5, 50, 5),
            "num_gen_updates": tune.randint(1, 10),
        }
    },
    "domain_disc": {"units": tune.sample_from(_get_disc_units)},
}

conditional_dann = {
    "training": {
        "approach": {
            "lr": tune.qloguniform(1e-5, 1e-2, 1e-5),
            "dann_factor": tune.qloguniform(0.1, 10.0, 0.1),
            "dynamic_adaptive_factor": tune.quniform(0.1, 0.9, 0.1),
        }
    },
    "domain_disc": {"units": tune.sample_from(_get_disc_units)},
}

conditional_mmd = {
    "training": {
        "approach": {
            "lr": tune.qloguniform(1e-5, 1e-2, 1e-5),
            "mmd_factor": tune.qloguniform(0.1, 10.0, 0.1),
            "num_mmd_kernels": tune.randint(1, 5),
            "dynamic_adaptive_factor": tune.quniform(0.1, 0.9, 0.1),
        }
    },
}

consistency = {
    "training": {
        "approach": {
            "lr": tune.qloguniform(1e-5, 1e-2, 1e-5),
            "consistency_factor": tune.quniform(0.1, 1.0, 0.1),
        }
    },
    "domain_disc": {"units": tune.sample_from(_get_disc_units)},
}


def _get_alpha(config):
    return config["training"]["approach"]["alpha_healthy"]


latent_align = {
    "training": {
        "approach": {
            "lr": tune.qloguniform(1e-5, 1e-2, 1e-5),
            "alpha_healthy": tune.qloguniform(0.1, 10.0, 0.1),
            "alpha_direction": tune.sample_from(_get_alpha),
            "alpha_level": tune.sample_from(_get_alpha),
            "alpha_fusion": tune.sample_from(_get_alpha),
        }
    },
}

mmd = {
    "training": {
        "approach": {
            "lr": tune.qloguniform(1e-5, 1e-2, 1e-5),
            "mmd_factor": tune.qloguniform(0.1, 10.0, 0.1),
            "num_mmd_kernels": tune.randint(1, 5),
        }
    },
}

pseudo_labels = {
    "training": {
        "approach": {
            "lr": tune.qloguniform(1e-5, 1e-2, 1e-5),
        },
    },
}
