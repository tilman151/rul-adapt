from typing import Callable, List, Literal, Tuple

import torch
import torchmetrics
from torch import nn

import rul_adapt


class AdaptionEvaluator(nn.Module):
    def __init__(
        self,
        network_func: Callable[[torch.Tensor], torch.Tensor],
        log_func: Callable[[str, torchmetrics.Metric], None],
        score_mode: Literal["phm08", "phm12"] = "phm08",
        degraded_only: bool = False,
    ):
        super().__init__()

        self.network_func = network_func
        self.log_func = log_func
        self.score_mode = score_mode
        self.degraded_only = degraded_only

        self.val_metrics = self._get_default_metrics()
        self.test_metrics = self._get_default_metrics()

    def _get_default_metrics(self) -> nn.ModuleDict:
        return nn.ModuleDict(
            {
                "source": nn.ModuleDict(
                    {
                        "rmse": torchmetrics.MeanSquaredError(squared=False),
                        "score": rul_adapt.loss.RULScore(self.score_mode),
                    }
                ),
                "target": nn.ModuleDict(
                    {
                        "rmse": torchmetrics.MeanSquaredError(squared=False),
                        "score": rul_adapt.loss.RULScore(self.score_mode),
                    }
                ),
            }
        )

    def validation(
        self, batch: List[torch.Tensor], domain: Literal["source", "target"]
    ) -> None:
        self._evaluate("val", self.val_metrics, batch, domain)

    def test(
        self, batch: List[torch.Tensor], domain: Literal["source", "target"]
    ) -> None:
        self._evaluate("test", self.test_metrics, batch, domain)

    def _evaluate(
        self,
        prefix: str,
        metrics: nn.ModuleDict,
        batch: List[torch.Tensor],
        domain: Literal["source", "target"],
    ) -> None:
        self._check_domain(domain, prefix)
        features, labels = batch
        features, labels = filter_batch(features, labels, self.degraded_only)
        labels = labels[:, None]
        predictions = self.network_func(features)
        for metric_name, metric in metrics[domain].items():
            metric(predictions, labels)
            self.log_func(f"{prefix}/{domain}/{metric_name}", metric)

    def _check_domain(self, domain: str, prefix: str) -> None:
        if domain not in ["source", "target"]:
            raise RuntimeError(
                f"Unexpected {prefix} domain '{domain}'. "
                "Use either 'source' or 'target'."
            )


def filter_batch(
    features: torch.Tensor, labels: torch.Tensor, degraded_only: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    if degraded_only:
        if torch.any(labels > 1.0):
            raise RuntimeError(
                "Degradation-only evaluation configured which works only with "
                "normalized RUL, but labels contain values greater than 1.0."
            )
        degraded = labels < 1.0
        features = features[degraded]
        labels = labels[degraded]

    return features, labels
