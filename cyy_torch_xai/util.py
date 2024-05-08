from cyy_torch_toolbox import MachineLearningPhase, Trainer
from cyy_torch_toolbox.typing import TensorDict


def get_test_gradient(trainer: Trainer) -> TensorDict:
    inferencer = trainer.get_inferencer(
        phase=MachineLearningPhase.Test, deepcopy_model=False
    )
    return inferencer.get_gradient()
