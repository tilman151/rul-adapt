from typing import Any, Dict

import hydra.utils

from rul_adapt.run import common


def dann(config: Dict[str, Any]):
    dm = common.get_adaption_datamodule(config)

    if common.should_be_pretrained(config):
        best_pretrained = common.run_pretraining(config, dm)
        feature_extractor = best_pretrained.feature_extractor
        regressor = best_pretrained.regressor
    else:
        feature_extractor, regressor = common.get_models(config)

    domain_disc = hydra.utils.instantiate(config["domain_disc"])
    approach = common.get_approach(config)
    approach.set_model(feature_extractor, regressor, domain_disc)
    result = common.run_adaption(config, approach, dm)

    return result
