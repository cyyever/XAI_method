from typing import Any

import torch.optim
from cyy_torch_toolbox import MachineLearningPhase

from ..config import DeterministicTrainingConfig
from .lean_hydra_hook import LeanHyDRAHook
from .lean_hydra_sgd_hook import LeanHyDRASGDHook


class LeanHyDRAConfig(DeterministicTrainingConfig):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.tracking_percentage: float | None = None

    def _create_hook(self, **hook_kwargs: Any) -> LeanHyDRAHook:
        assert self.deterministic_training.last_trainer is not None
        optimizer = self.deterministic_training.last_trainer.get_optimizer()
        test_gradient = hook_kwargs.get("test_gradient")
        if test_gradient is None:
            tester = self.deterministic_training.last_trainer.get_inferencer(
                phase=MachineLearningPhase.Test, deepcopy_model=False
            )
            test_gradient = tester.get_gradient()
            del tester
        assert test_gradient is not None
        match optimizer:
            case torch.optim.SGD():
                hydra_hook = LeanHyDRASGDHook(
                    test_gradient=test_gradient, use_hessian=False
                )
            case _:
                raise NotImplementedError(f"Unsupported optimizer {type(optimizer)}")
        return hydra_hook
