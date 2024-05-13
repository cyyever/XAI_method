from typing import Any

import torch
from cyy_naive_lib.log import get_logger
from cyy_torch_algorithm.computation import BatchHVPHook, SampleGradientHook
from cyy_torch_toolbox import (Hook, IndicesType, ModelUtil, OptionalTensor,
                               get_device)

from ..typing import SampleContributionDict


class BaseHook(Hook):
    def __init__(self, use_hessian: bool = False) -> None:
        super().__init__(stripable=True)
        self._use_hessian: bool = use_hessian
        self._batch_hvp_hook: None | BatchHVPHook = None
        if self._use_hessian:
            self._batch_hvp_hook = BatchHVPHook()

        self._sample_gradient_hook: SampleGradientHook = SampleGradientHook()
        self.__computed_indices: set[int] | None = None
        self._contributions: OptionalTensor = None
        self._training_set_size: int | None = None

    @property
    def batch_hvp_hook(self) -> BatchHVPHook:
        assert self._batch_hvp_hook is not None
        return self._batch_hvp_hook

    @property
    def training_set_size(self) -> int:
        assert self._training_set_size is not None
        return self._training_set_size

    @property
    def use_hessian(self) -> bool:
        return self._use_hessian

    @property
    def contributions(self) -> torch.Tensor:
        assert self._contributions is not None
        return self._contributions

    def _before_execute(self, **kwargs: Any) -> None:
        if "executor" in kwargs:
            trainer = kwargs["executor"]
            self._training_set_size = trainer.dataset_size
            device = trainer.device
        else:
            self._training_set_size = kwargs["training_set_size"]
            device = get_device()

        if self.__computed_indices is None:
            self.set_computed_indices(range(self.training_set_size))
        else:
            get_logger().info("only compute %s indices", len(self.computed_indices))
        self._contributions = torch.zeros(self.training_set_size).to(
            device, non_blocking=True
        )

    @property
    def computed_indices(self) -> set[int]:
        assert self.__computed_indices is not None
        return self.__computed_indices

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

    def set_computed_indices(self, computed_indices: IndicesType) -> None:
        assert computed_indices
        self.__computed_indices = set(computed_indices)
        self._sample_gradient_hook.set_computed_indices(computed_indices)

    @property
    def contribution_dict(self) -> SampleContributionDict:
        return {idx: self.contributions[idx].item() for idx in self.computed_indices}

    def _after_execute(self, **kwargs: Any) -> None:
        assert self._contributions is not None
        assert self._contributions.shape[0] == self.training_set_size
        self._sample_gradient_hook.release_queue()
        if self._batch_hvp_hook is not None:
            self._batch_hvp_hook.release_queue()
