from rul_adapt import tune


def test__flatten_dict():
    d = {
        "training": {"approach": {"lr": 0.1, "dann_factor": 2.0}},
        "domain_disc": {"units": [32, 1]},
    }
    flat = tune._flatten_dict(d)

    exp_flat = [
        "training.approach.lr=0.1",
        "training.approach.dann_factor=2.0",
        "domain_disc.units=[32,1]",
    ]
    assert flat == exp_flat
