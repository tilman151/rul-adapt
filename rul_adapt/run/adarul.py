from typing import Any, Dict

import hydra.utils

from rul_adapt.run import common


def adarul(config: Dict[str, Any]):
    dm = common.get_adaption_datamodule(config)
    best_pretrained = common.run_pretraining(config, dm)

    domain_disc = hydra.utils.instantiate(config["domain_disc"])
    approach = common.get_approach(config)
    approach.set_model(
        best_pretrained.feature_extractor, best_pretrained.regressor, domain_disc
    )
    result = common.run_adaption(config, approach, dm)

    return result
