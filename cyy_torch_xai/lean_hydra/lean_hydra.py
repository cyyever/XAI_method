from typing import Any

import torch
from cyy_torch_toolbox import (ExecutorHookPoint, HookCollection, IndicesType,
                               ModelEvaluator, ModelGradient)

from ..typing import SampleContributions
from .lean_hydra_sgd_hook import LeanHyDRASGDHook


class LeanHyDRA:
    def __init__(
        self,
        model_evaluator: ModelEvaluator,
        optimizer: torch.optim.Optimizer,
        test_gradient: ModelGradient,
        training_set_size: int,
    ) -> None:
        self.__hooks = HookCollection()
        match optimizer:
            case torch.optim.SGD():
                self._hydra_hook = LeanHyDRASGDHook(test_gradient=test_gradient)
            case _:
                raise RuntimeError(f"unsupported optimizer type {type(optimizer)}")
        self.__hooks.append_hook(self._hydra_hook)
        self.optimizer = optimizer
        self.model_evaluator = model_evaluator
        self.__hooks.exec_hooks(
            hook_point=ExecutorHookPoint.BEFORE_EXECUTE,
            training_set_size=training_set_size,
        )
        self.__end_exe: bool = False

    def set_computed_indices(self, computed_indices: IndicesType) -> None:
        self._hydra_hook.set_computed_indices(computed_indices=computed_indices)

    def iterate(self, sample_indices, inputs, targets, **kwargs) -> None:
        self.__hooks.exec_hooks(
            hook_point=ExecutorHookPoint.BEFORE_BATCH,
            model_evaluator=self.model_evaluator,
            sample_indices=sample_indices,
            inputs=inputs,
            targets=targets,
            executor=None,
        )
        self.__hooks.exec_hooks(
            hook_point=ExecutorHookPoint.AFTER_BATCH,
            batch_size=len(sample_indices),
            step_skipped=False,
            model_evaluator=self.model_evaluator,
            optimizer=self.optimizer,
        )

    def get_contribution(self, **kwargs: Any) -> SampleContributions:
        if not self.__end_exe:
            self.__hooks.exec_hooks(hook_point=ExecutorHookPoint.AFTER_EXECUTE)
            self.__end_exe = True
        return self._hydra_hook.contribution_dict
