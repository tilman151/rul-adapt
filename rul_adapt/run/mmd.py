from typing import Any, Dict

import hydra.utils

from rul_adapt.run import common


def mmd(config: Dict[str, Any]):
    dm = common.get_adaption_datamodule(config)
    feature_extractor, regressor = common.get_models(config)
    approach = common.get_approach(config)
    approach.set_model(feature_extractor, regressor)
    result = common.run_adaption(config, approach, dm)

    return result
