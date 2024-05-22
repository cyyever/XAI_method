import torch

from cyy_naive_lib.log import log_error
from cyy_torch_toolbox import (IndicesType, MachineLearningPhase,
                               OptionalIndicesType, Trainer)

from .evaluator import OutputFeatureModelEvaluator
from ..contribution import SubsetContribution


def __get_output_features(
    trainer: Trainer, phase: MachineLearningPhase, sample_indices: OptionalIndicesType
) -> dict[int, torch.Tensor]:
    inferencer = trainer.get_inferencer(phase=phase, deepcopy_model=True)
    inferencer.replace_model_evaluator(
        lambda model_evaluator: OutputFeatureModelEvaluator(evaluator=model_evaluator)
    )
    if sample_indices is not None:
        inferencer.mutable_dataset_collection.set_subset(
            phase=phase, indices=set(sample_indices)
        )
    inferencer.inference()
    assert isinstance(inferencer.model_evaluator, OutputFeatureModelEvaluator)
    if sample_indices is not None:
        assert len(inferencer.model_evaluator.output_features) == len(
            set(sample_indices)
        )
    return inferencer.model_evaluator.output_features


def compute_representer_point_values(
    trainer: Trainer,
    test_indices: IndicesType,
    training_indices: OptionalIndicesType = None,
) -> SubsetContribution:
    test_features = __get_output_features(
        trainer=trainer,
        phase=MachineLearningPhase.Test,
        sample_indices=test_indices,
    )

    training_features = __get_output_features(
        trainer=trainer,
        phase=MachineLearningPhase.Training,
        sample_indices=training_indices,
    )
    log_error("aaa %s", len(test_features))
    log_error("vvv %s", len(training_features))
    contribution = SubsetContribution()
    contribution.set_tracked_indices(list(training_features.keys()))
    return contribution
