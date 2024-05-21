from typing import Any

from cyy_torch_algorithm.computation import BatchHVPHook

from ..base_hook import SampleGradientXAIHook


class BaseHook(SampleGradientXAIHook):
    def __init__(self, use_hessian: bool = False) -> None:
        super().__init__()
        self._use_hessian: bool = use_hessian
        self._batch_hvp_hook: None | BatchHVPHook = None
        if self._use_hessian:
            self._batch_hvp_hook = BatchHVPHook()

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

    def _after_execute(self, **kwargs: Any) -> None:
        if self._batch_hvp_hook is not None:
            self._batch_hvp_hook.release()
        super()._after_execute(**kwargs)
