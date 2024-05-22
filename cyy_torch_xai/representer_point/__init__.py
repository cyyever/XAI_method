import copy

from cyy_naive_lib.log import log_error
from cyy_torch_toolbox import (IndicesType, Inferencer, MachineLearningPhase,
                               OptionalIndicesType, Trainer)

from ..typing import SampleContributions
from .evaluator import OutputFeatureModelEvaluator


def __get_inferencer(
    trainer: Trainer, phase: MachineLearningPhase, sample_indices: OptionalIndicesType
) -> Inferencer:
    inferencer = trainer.get_inferencer(phase=phase, deepcopy_model=True)
    inferencer.replace_model_evaluator(
        lambda model_evaluator: OutputFeatureModelEvaluator(evaluator=model_evaluator)
    )
    if sample_indices is not None:
        inferencer.mutable_dataset_collection.set_subset(
            phase=phase, indices=set(sample_indices)
        )
    return inferencer


def compute_representer_point_values(
    trainer: Trainer,
    test_indices: IndicesType,
    training_indices: OptionalIndicesType = None,
) -> list[SampleContributions]:
    inferencer = __get_inferencer(
        trainer=trainer,
        phase=MachineLearningPhase.Test,
        sample_indices=test_indices,
    )
    inferencer.inference()
    assert isinstance(inferencer.model_evaluator, OutputFeatureModelEvaluator)
    assert len(inferencer.model_evaluator.output_features) == len(
        set(test_indices)
    )

    inferencer = __get_inferencer(
        trainer=trainer,
        phase=MachineLearningPhase.Training,
        sample_indices=training_indices,
    )
    inferencer.inference()
    assert isinstance(inferencer.model_evaluator, OutputFeatureModelEvaluator)
    assert training_indices is None or len(
        inferencer.model_evaluator.output_features
    ) == len(set(training_indices))

    # training_inferencer = trainer.get_inferencer(
    #     phase=MachineLearningPhase.Training, deepcopy_model=True
    # )
    log_error("aaa %s", len(inferencer.model_evaluator.output_features))
    log_error("bbbb %s", len(set(test_indices)))
    return []
