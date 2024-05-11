from typing import Any

import torch
from cyy_naive_lib.log import get_logger
from cyy_torch_algorithm.computation.batch_hvp.batch_hvp_hook import \
    BatchHVPHook
from cyy_torch_algorithm.computation.sample_gradient.sample_gradient_hook import \
    SampleGradientHook
from cyy_torch_toolbox import ModelUtil
from cyy_torch_toolbox.device import get_device
from cyy_torch_toolbox.hook import Hook
from cyy_torch_toolbox.typing import IndicesType


class BaseHook(Hook):
    def __init__(self, use_hessian: bool = False) -> None:
        super().__init__(stripable=True)
        self._use_hessian = use_hessian
        self._batch_hvp_hook = None
        if self._use_hessian:
            self._batch_hvp_hook = BatchHVPHook()

        self._sample_gradient_hook = SampleGradientHook()
        self._computed_indices: set[int] | None = None
        self._contributions: torch.Tensor | None = None
        self._training_set_size: int | None = None

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

        assert self._training_set_size is not None
        if self.computed_indices is None:
            self._computed_indices = set(range(self._training_set_size))
        else:
            get_logger().info("only compute %s indices", len(self.computed_indices))
        self._contributions = torch.zeros(self._training_set_size).to(
            device, non_blocking=True
        )

    @property
    def computed_indices(self) -> set[int] | None:
        return self._computed_indices

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
        self._computed_indices = set(computed_indices)
        self._sample_gradient_hook.set_computed_indices(computed_indices)

    def _after_execute(self, **kwargs: Any) -> None:
        assert self._contributions is not None
        assert self._contributions.shape[0] == self._training_set_size
        self._sample_gradient_hook.release_queue()
        if self._batch_hvp_hook is not None:
            self._batch_hvp_hook.release_queue()
