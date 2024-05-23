
import torch.optim
from cyy_torch_toolbox import (MachineLearningPhase, ModelGradient)

from .lean_hydra_sgd_hook import LeanHyDRASGDHook
from .lean_hydra_hook import LeanHyDRAHook
from ..config import DeterministicTrainingConfig


class LeanHyDRAConfig(DeterministicTrainingConfig):

    def _create_hook(self, test_gradient: ModelGradient) -> LeanHyDRAHook:
        assert self.deterministic_training.last_trainer is not None
        optimizer = self.deterministic_training.last_trainer.get_optimizer()
        match optimizer:
            case torch.optim.SGD():
                hydra_hook = LeanHyDRASGDHook(
                    test_gradient=test_gradient, use_hessian=False
                )
            case _:
                raise NotImplementedError(f"Unsupported optimizer {type(optimizer)}")
        return hydra_hook

    def recreate_trainer_and_hook(self, test_gradient: None | ModelGradient = None) -> dict:
        assert self.deterministic_training.last_trainer is not None
        if test_gradient is None:
            tester = self.deterministic_training.last_trainer.get_inferencer(
                phase=MachineLearningPhase.Test, deepcopy_model=False
            )
            test_gradient = tester.get_gradient()
            del tester
        hydra_hook = self._create_hook(test_gradient=test_gradient)
        trainer = self.deterministic_training.recreate_trainer()
        trainer.append_hook(hydra_hook)
        return {"trainer": trainer, "hook": hydra_hook, "test_gradient": test_gradient}
