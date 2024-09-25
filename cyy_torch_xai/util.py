from cyy_torch_toolbox import MachineLearningPhase, ModelGradient, Trainer


def get_test_gradient(trainer: Trainer) -> ModelGradient:
    inferencer = trainer.get_inferencer(
        phase=MachineLearningPhase.Test, deepcopy_model=False
    )
    return inferencer.get_gradient()
