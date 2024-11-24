from typing import Any

import torch
from cyy_naive_lib.log import log_info
from cyy_torch_algorithm.computation import SampleGradientHook
from cyy_torch_toolbox import Hook, IndicesType, ModelUtil, Trainer

from .contribution import SubsetContribution


class SampleXAIHook(Hook):
    def __init__(self) -> None:
        super().__init__(stripable=True)
        self.__has_tracked_samples: bool = False
        self.__training_set_size: int | None = None
        self._contribution: SubsetContribution = SubsetContribution()

    @property
    def computed_indices(self) -> set[int]:
        return self._contribution.tracked_indices

    def set_computed_indices(self, computed_indices: IndicesType) -> None:
        assert computed_indices
        self.__has_tracked_samples = True
        self._contribution.set_tracked_indices(computed_indices)

    @property
    def training_set_size(self) -> int:
        assert self.__training_set_size is not None
        return self.__training_set_size

    def _before_execute(self, **kwargs: Any) -> None:
        if "executor" in kwargs:
            trainer = kwargs["executor"]
            assert isinstance(trainer, Trainer)
            self.__training_set_size = trainer.dataset_size
        else:
            self.__training_set_size = kwargs["training_set_size"]

        if not self.__has_tracked_samples:
            self.set_computed_indices(range(self.training_set_size))
        else:
            log_info("only compute %s indices", len(self.computed_indices))
        self._contribution.clear_contributions()


class SampleGradientXAIHook(SampleXAIHook):
    def __init__(self) -> None:
        super().__init__()
        self._sample_gradient_hook: SampleGradientHook = SampleGradientHook()

    def set_computed_indices(self, computed_indices: IndicesType) -> None:
        super().set_computed_indices(computed_indices=computed_indices)
        self._sample_gradient_hook.set_computed_indices(self.computed_indices)

    def _get_optimizer(self, **kwargs) -> torch.optim.Optimizer:
        if "executor" in kwargs:
            trainer = kwargs["executor"]
            return trainer.get_optimizer()
        return kwargs["optimizer"]

    def _get_model_util(self, **kwargs: Any) -> ModelUtil:
        if "executor" in kwargs:
            trainer = kwargs["executor"]
            return trainer.model_util
        return kwargs["model_evaluator"].model_util

    def _after_execute(self, **kwargs: Any) -> None:
        self._sample_gradient_hook.release()
