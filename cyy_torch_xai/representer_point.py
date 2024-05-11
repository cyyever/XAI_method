
from cyy_torch_toolbox import MachineLearningPhase, Trainer
from cyy_torch_toolbox.typing import ModelGradient, OptionalIndicesType

from .util import get_test_gradient


def compute_representer_point_values(
    trainer: Trainer,
    training_indices: OptionalIndicesType,
    test_gradient: ModelGradient | None = None,
) -> dict[int, float]:
    if test_gradient is None:
        test_gradient = get_test_gradient(trainer=trainer)

    inferencer = trainer.get_inferencer(
        phase=MachineLearningPhase.Training, deepcopy_model=True
    )
    return {}
