from rul_adapt import tune


def test_config2overrides():
    d = {
        "training": {"approach": {"lr": 0.1, "dann_factor": 2.0}},
        "domain_disc": {"units": [32, 1]},
    }
    flat = tune.config2overrides(d)

    exp_flat = [
        "training.approach.lr=0.1",
        "training.approach.dann_factor=2.0",
        "domain_disc.units=[32,1]",
    ]
    assert flat == exp_flat
