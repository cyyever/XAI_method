from collections.abc import Callable
from typing import Any

from cyy_torch_algorithm.retraining import DeterministicTraining
from cyy_torch_toolbox import Config, Trainer


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

    def _create_hook(self, **hook_kwargs: Any) -> Any:
        raise NotImplementedError()

    def recreate_trainer_and_hook(self, **hook_kwargs: Any) -> dict:
        assert self.deterministic_training.last_trainer is not None
        hydra_hook = self._create_hook(**hook_kwargs)
        trainer = self.deterministic_training.recreate_trainer()
        trainer.append_hook(hydra_hook)
        return {"trainer": trainer, "hook": hydra_hook}
