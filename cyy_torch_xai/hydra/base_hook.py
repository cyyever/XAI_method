from typing import Any

import torch
import torch.optim
from cyy_torch_algorithm.computation import BatchHVPHook, SampleGradientHook
from cyy_torch_toolbox import IndicesType

from ..base_hook import SampleBaseHook
from ..typing import SampleContributions


class BaseHook(SampleBaseHook):
    def __init__(self, use_hessian: bool = False) -> None:
        super().__init__()
        self._use_hessian: bool = use_hessian
        self._batch_hvp_hook: None | BatchHVPHook = None
        if self._use_hessian:
            self._batch_hvp_hook = BatchHVPHook()

        self._sample_gradient_hook: SampleGradientHook = SampleGradientHook()

    @property
    def batch_hvp_hook(self) -> BatchHVPHook:
        assert self._batch_hvp_hook is not None
        return self._batch_hvp_hook

    @property
    def use_hessian(self) -> bool:
        return self._use_hessian


    # def _get_model_util(self, **kwargs: Any) -> ModelUtil:
    #     if "executor" in kwargs:
    #         trainer = kwargs["executor"]
    #         return trainer.model_util
    #     return kwargs["model_evaluator"].model_util

    def set_computed_indices(self, computed_indices: IndicesType) -> None:
        super().set_computed_indices(computed_indices)
        self._sample_gradient_hook.set_computed_indices(computed_indices)

    @property
    def contribution_dict(self) -> SampleContributions:
        return {idx: self.contributions[idx].item() for idx in self.computed_indices}

    def _after_execute(self, **kwargs: Any) -> None:
        assert self.contributions.shape[0] == self.training_set_size
        self._sample_gradient_hook.release_queue()
        if self._batch_hvp_hook is not None:
            self._batch_hvp_hook.release_queue()
