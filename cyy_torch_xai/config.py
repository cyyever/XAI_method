from typing import Callable

from cyy_torch_algorithm.retraining import DeterministicTraining
from cyy_torch_toolbox import (Config, Trainer)



class DeterministicTrainingConfig(Config):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.deterministic_training = DeterministicTraining(self)

    def create_deterministic_trainer(
        self, trainer_fun: None | Callable = None
    ) -> Trainer:
        return self.deterministic_training.create_deterministic_trainer(
            trainer_fun=trainer_fun
        )
