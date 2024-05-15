import copy

from cyy_torch_toolbox import (IndicesType, Inferencer, MachineLearningPhase,
                               OptionalIndicesType, Trainer)

from .evaluator import OutputFeatureModelEvaluator


def __get_inferencer(
    trainer: Trainer, phase: MachineLearningPhase, sample_indices: OptionalIndicesType
) -> Inferencer:
    test_inferencer = trainer.get_inferencer(phase=phase, deepcopy_model=True)
    test_inferencer.replace_model_evaluator(
        lambda model_evaluator: OutputFeatureModelEvaluator(evaluator=model_evaluator)
    )
    if sample_indices is not None:
        test_inferencer.dc.set_subset(phase=phase, indices=sample_indices)

    return test_inferencer


def compute_representer_point_values(
    trainer: Trainer,
    test_indices: IndicesType,
    training_indices: OptionalIndicesType = None,
) -> dict[int, float]:
    trainer = copy.deepcopy(trainer)
    test_inferencer = __get_inferencer(
        trainer=trainer,
        phase=MachineLearningPhase.Training,
        sample_indices=training_indices,
    )
    test_inferencer.inference()
    assert isinstance(test_inferencer.model_evaluator, OutputFeatureModelEvaluator)
    assert len(test_inferencer.model_evaluator.output_features) == len(test_indices)

    # training_inferencer = trainer.get_inferencer(
    #     phase=MachineLearningPhase.Training, deepcopy_model=True
    # )
    return {}
