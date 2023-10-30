from typing import Callable

import torch.optim
from cyy_torch_algorithm.retraining import DeterministicTraining
from cyy_torch_toolbox.default_config import DefaultConfig
from cyy_torch_toolbox.ml_type import MachineLearningPhase
from cyy_torch_toolbox.trainer import Trainer

from .lean_hydra_sgd_hook import LeanHyDRASGDHook


class LeanHyDRAConfig(DefaultConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.deterministic_training = DeterministicTraining(self)

    def create_deterministic_trainer(
        self, trainer_fun: None | Callable = None
    ) -> Trainer:
        return self.deterministic_training.create_deterministic_trainer(
            trainer_fun=trainer_fun
        )

    def recreate_trainer_and_hook(self, test_gradient=None) -> tuple:
        assert self.deterministic_training.last_trainer is not None
        if test_gradient is None:
            tester = self.deterministic_training.last_trainer.get_inferencer(
                phase=MachineLearningPhase.Test, deepcopy_model=False
            )
            test_gradient = tester.get_gradient()
            del tester
        optimizer = self.deterministic_training.last_trainer.get_optimizer()
        match optimizer:
            case torch.optim.SGD():
                hydra_hook = LeanHyDRASGDHook(test_gradient=test_gradient)
            case _:
                raise NotImplementedError(f"Unsupported optimizer {type(optimizer)}")
        trainer = self.deterministic_training.recreate_trainer()
        trainer.append_hook(hydra_hook)

        return trainer, hydra_hook, test_gradient
