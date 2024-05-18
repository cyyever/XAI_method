from cyy_torch_toolbox import MachineLearningPhase, Trainer
from cyy_torch_toolbox.typing import ModelGradient


def get_test_gradient(trainer: Trainer) -> ModelGradient:
    inferencer = trainer.get_inferencer(
        phase=MachineLearningPhase.Test, deepcopy_model=False
    )
    return inferencer.get_gradient()
