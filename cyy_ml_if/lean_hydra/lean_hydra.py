import torch
from cyy_torch_toolbox.hook import HookCollection
from cyy_torch_toolbox.ml_type import ExecutorHookPoint
from cyy_torch_toolbox.model_evaluator import ModelEvaluator

from .lean_hydra_adam_hook import LeanHyDRAAdamHook
from .lean_hydra_sgd_hook import LeanHyDRASGDHook


class LeanHyDRA:
    def __init__(
        self,
        model,
        loss_function,
        optimizer,
        test_gradient,
        training_set_size,
    ):
        self.__hooks = HookCollection()
        match optimizer:
            case torch.optim.SGD():
                self._hydra_hook = LeanHyDRASGDHook(test_gradient=test_gradient)
            case torch.optim.Adam():
                self._hydra_hook = LeanHyDRAAdamHook(test_gradient=test_gradient)
            case _:
                raise RuntimeError(f"unsupported optimizer type {type(optimizer)}")
        self.__hooks.append_hook(self._hydra_hook)
        self.optimizer = optimizer
        self.model_evaluator = ModelEvaluator(model, loss_function)
        self.__hooks.exec_hooks(
            hook_point=ExecutorHookPoint.BEFORE_EXECUTE,
            training_set_size=training_set_size,
        )
        self.__end_exe: bool = False

    def set_computed_indices(self, computed_indices):
        self._hydra_hook.set_computed_indices(computed_indices=computed_indices)

    def iterate(self, sample_indices, inputs, targets, **kwargs):
        self.__hooks.exec_hooks(
            hook_point=ExecutorHookPoint.AFTER_FORWARD,
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

    def cancel_forward(self, **kwargs):
        self.__hooks.exec_hooks(
            hook_point=ExecutorHookPoint.CANCEL_FORWARD,
        )

    def get_contribution(self, **kwargs):
        if not self.__end_exe:
            self.__hooks.exec_hooks(hook_point=ExecutorHookPoint.AFTER_EXECUTE)
            self.__end_exe = True
            return {
                idx: self._hydra_hook.contributions[idx].item()
                for idx in self._hydra_hook.computed_indices
            }
        return self._hydra_hook.contributions
