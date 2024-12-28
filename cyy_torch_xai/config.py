from collections.abc import Callable
from typing import Any

from cyy_torch_algorithm.retraining import DeterministicTraining
from cyy_torch_toolbox import Config, Hook, Trainer


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

    def _create_hook(self, **hook_kwargs: Any) -> Hook:
        raise NotImplementedError()

    def recreate_trainer_and_hook(self, **hook_kwargs: Any) -> tuple[Trainer, Hook]:
        assert self.deterministic_training.last_trainer is not None
        hydra_hook = self._create_hook(**hook_kwargs)
        trainer = self.deterministic_training.recreate_trainer()
        trainer.append_hook(hook=hydra_hook)
        return trainer, hydra_hook
