"""The Adversarial Domain Adaption for Remaining Useful Life (ADARUL) approach
pre-trains a feature extractor and regressor on the source domain in a supervised
fashion. Afterwards the feature extractor is adapted by feeding it the target
features and training it adversarial against a domain discriminator. The
discriminator is trained to distinguish the source features fed to a frozen version
of the pre-trained feature extractor and the target features fed to the adapted
feature extractor.

The approach was first introduced by [Ragab et al.](
https://doi.org/10.1109/ICPHM49022.2020.9187053) and evaluated on the CMAPSS dataset."""

import copy
from typing import Optional, Any, List, Dict, Literal

import torch
from torch import nn

from rul_adapt import utils
from rul_adapt.approach.abstract import AdaptionApproach
from rul_adapt.approach.evaluation import AdaptionEvaluator
from rul_adapt.model import FullyConnectedHead


class AdaRulApproach(AdaptionApproach):
    """The ADARUL approach uses a GAN setup to adapt a feature extractor. This
    approach should only be used with a pre-trained feature extractor.

    The regressor and domain discriminator need the same number of input units as the
    feature extractor has output units. The discriminator is not allowed to have an
    activation function on its last layer for it to work with its loss.

    Examples:
        >>> from rul_adapt import model
        >>> from rul_adapt import approach
        >>> feat_ex = model.CnnExtractor(1, [16, 16, 1], 10, fc_units=16)
        >>> reg = model.FullyConnectedHead(16, [1])
        >>> disc = model.FullyConnectedHead(16, [8, 1], act_func_on_last_layer=False)
        >>> pre = approach.SupervisedApproach("mse", 125, lr=0.001)
        >>> pre.set_model(feat_ex, reg)
        >>> main = approach.AdaRulApproach(5, 1, 125, lr=0.001)
        >>> main.set_model(pre.feature_extractor, pre.regressor, disc)
    """

    CHECKPOINT_MODELS = ["_domain_disc", "frozen_feature_extractor"]

    _domain_disc: nn.Module
    frozen_feature_extractor: nn.Module

    def __init__(
        self,
        num_disc_updates: int,
        num_gen_updates: int,
        max_rul: Optional[int] = None,
        rul_score_mode: Literal["phm08", "phm12"] = "phm08",
        evaluate_degraded_only: bool = False,
        **optim_kwargs: Any,
    ) -> None:
        """
        Create a new ADARUL approach.

        The discriminator is first trained for `num_disc_updates` batches.
        Afterward, the feature extractor (generator) is trained for
        `num_gen_updates`. This cycle repeats until the epoch ends.

        The regressor is supposed to output a value between [0, 1] which is then
        scaled by `max_rul`.

        For more information about the possible optimizer keyword arguments,
        see [here][rul_adapt.utils.OptimizerFactory].

        Args:
            max_rul: Maximum RUL value of the training data.
            num_disc_updates: Number of batches to update discriminator with.
            num_gen_updates: Number of batches to update generator with.
            rul_score_mode: The mode for the val and test RUL score, either 'phm08'
                            or 'phm12'.
            evaluate_degraded_only: Whether to only evaluate the RUL score on degraded
                                    samples.
            **optim_kwargs: Keyword arguments for the optimizer, e.g. learning rate.
        """
        super().__init__()

        self.automatic_optimization = False  # use manual optimization loop

        self.num_disc_updates = num_disc_updates
        self.num_gen_updates = num_gen_updates
        self.max_rul = max_rul
        self.rul_score_mode = rul_score_mode
        self.evaluate_degraded_only = evaluate_degraded_only
        self.optim_kwargs = optim_kwargs

        self._disc_counter, self._gen_counter = 0, 0
        self._get_optimizer = utils.OptimizerFactory(**self.optim_kwargs)

        self.gan_loss = nn.BCEWithLogitsLoss()

        self.evaluator = AdaptionEvaluator(
            self.forward, self.log, self.rul_score_mode, self.evaluate_degraded_only
        )

        self.save_hyperparameters()

    def set_model(
        self,
        feature_extractor: nn.Module,
        regressor: nn.Module,
        domain_disc: Optional[nn.Module] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Set the feature extractor, regressor and domain discriminator for this approach.
        The discriminator is not allowed to have an activation function on its last
        layer and needs to use only a single output neuron.

        A frozen copy of the feature extractor is produced to be used for the *real*
        samples fed to the discriminator. The feature extractor should, therefore,
        be pre-trained.

        Args:
            feature_extractor: The feature extraction network.
            regressor: The RUL regression network.
            domain_disc: The domain discriminator network.
        """
        domain_disc = self._check_domain_disc(domain_disc)
        super().set_model(feature_extractor, regressor, *args, **kwargs)
        self._domain_disc = domain_disc
        self.frozen_feature_extractor = copy.deepcopy(feature_extractor)
        self.frozen_feature_extractor.requires_grad_(False)  # freeze network
        self.log_model_hyperparameters("_domain_disc")

    def _check_domain_disc(self, domain_disc: Optional[nn.Module]) -> nn.Module:
        if domain_disc is None:
            raise ValueError(
                "No domain discriminator was set. This approach is unlikely to work."
            )
        if (
            isinstance(domain_disc, FullyConnectedHead)
            and domain_disc.act_func_on_last_layer
        ):
            raise ValueError(
                "Domain discriminator has an activation function on its last layer. "
                "This is not allowed due to torch.nn.BCEWithLogitsLoss being used as "
                "its loss. Please set 'act_func_on_last_layer' to False."
            )

        return domain_disc

    @property
    def domain_disc(self):
        """The domain discriminator network."""
        if hasattr(self, "_domain_disc"):
            return self._domain_disc
        else:
            raise RuntimeError("Domain disc used before 'set_model' was called.")

    def configure_optimizers(self) -> List[Dict[str, Any]]:
        """Configure an optimizer for the generator and discriminator respectively."""
        return [
            self._get_optimizer(self.domain_disc.parameters()),
            self._get_optimizer(self.feature_extractor.parameters()),
        ]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Predict the RUL values for a batch of input features."""
        max_rul = self.max_rul or 1
        pred = self.regressor(self.feature_extractor(inputs)) * max_rul

        return pred

    def on_train_epoch_start(self) -> None:
        self._reset_update_counters()

    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Execute one training step.

        The `batch` argument is a list of three tensors representing the source
        features, source labels and target features. Each iteration either only the
        discriminator or only the generator is trained. The respective loss is logged.

        The *real* samples are source features passed though the frozen version of
        the feature extractor. The *fake* samples are the target features passed
        through the adapted feature extractor. The discriminator predicts if a sample
        came from the source or target domain.

        Args:
            batch: A list of a source feature, source label and target feature tensors.
            batch_idx: The index of the current batch.
        Returns:
            Either the discriminator or generator loss.
        """
        source, _, target = batch

        if self._updates_done():
            self._reset_update_counters()

        if self._should_update_disc():
            optim, _ = self.optimizers()  # type: ignore[misc]
            loss = self._get_disc_loss(source, target)
            self.log("train/disc_loss", loss)
            self._disc_counter += 1
        elif self._should_update_gen():
            _, optim = self.optimizers()  # type: ignore[misc]
            loss = self._get_gen_loss(target)
            self.log("train/gen_loss", loss)
            self._gen_counter += 1
        else:
            raise RuntimeError("Configuration error. Did update neither disc nor gen.")

        optim.zero_grad()  # type: ignore[union-attr]
        self.manual_backward(loss)
        optim.step()

        return loss

    def _should_update_disc(self):
        return self._disc_counter < self.num_disc_updates and self._gen_counter == 0

    def _should_update_gen(self):
        return (
            self._disc_counter == self.num_disc_updates
            and self._gen_counter < self.num_gen_updates
        )

    def _reset_update_counters(self):
        self._disc_counter, self._gen_counter = 0, 0

    def _updates_done(self) -> bool:
        return (
            self._disc_counter == self.num_disc_updates
            and self._gen_counter == self.num_gen_updates
        )

    def _get_disc_loss(self, source, target):
        batch_size = source.shape[0]
        source = self.frozen_feature_extractor(source).detach()
        target = self.feature_extractor(target).detach()
        domain_pred = self.domain_disc(torch.cat([source, target]))
        domain_labels = torch.cat(
            [
                torch.ones(batch_size, 1, device=self.device),  # real labels
                torch.zeros(batch_size, 1, device=self.device),  # fake labels
            ]
        )
        loss = self.gan_loss(domain_pred, domain_labels)

        return loss

    def _get_gen_loss(self, target):
        batch_size = target.shape[0]
        target = self.feature_extractor(target)
        domain_pred = self.domain_disc(target)
        domain_labels = torch.ones(batch_size, 1, device=self.device)  # flipped labels
        loss = self.gan_loss(domain_pred, domain_labels)

        return loss

    def validation_step(
        self, batch: List[torch.Tensor], batch_idx: int, dataloader_idx: int
    ) -> None:
        """
        Execute one validation step.

        The `batch` argument is a list of two tensors representing features and
        labels. A RUL prediction is made from the features and the validation RMSE
        and RUL score are calculated. The metrics recorded for `dataloader_idx` zero
        are assumed to be from the source domain and for `dataloader_idx` one from the
        target domain. The metrics are written to the configured logger under the
        prefix `val`.

        Args:
            batch: A list containing a feature and a label tensor.
            batch_idx: The index of the current batch.
            dataloader_idx: The index of the current dataloader (0: source, 1: target).
        """
        domain = utils.dataloader2domain(dataloader_idx)
        self.evaluator.validation(batch, domain)

    def test_step(
        self, batch: List[torch.Tensor], batch_idx: int, dataloader_idx: int
    ) -> None:
        """
        Execute one test step.

        The `batch` argument is a list of two tensors representing features and
        labels. A RUL prediction is made from the features and the validation RMSE
        and RUL score are calculated. The metrics recorded for `dataloader_idx` zero
        are assumed to be from the source domain and for `dataloader_idx` one from the
        target domain. The metrics are written to the configured logger under the
        prefix `test`.

        Args:
            batch: A list containing a feature and a label tensor.
            batch_idx: The index of the current batch.
            dataloader_idx: The index of the current dataloader (0: source, 1: target).
        """
        domain = utils.dataloader2domain(dataloader_idx)
        self.evaluator.test(batch, domain)
