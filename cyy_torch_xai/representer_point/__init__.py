import math

import torch
from cyy_torch_toolbox import (
    EvaluationMode,
    IndicesType,
    MachineLearningPhase,
    OptionalIndicesType,
    Trainer,
)
from cyy_torch_toolbox.tensor import dot_product

from ..contribution import SubsetContribution
from .evaluator import (
    OutputFeatureModelEvaluator,
    OutputGradientEvaluator,
    OutputModelEvaluator,
)


def __get_output(
    trainer: Trainer, phase: MachineLearningPhase, sample_indices: OptionalIndicesType
) -> dict[str, dict[int, torch.Tensor]]:
    inferencer = trainer.get_inferencer(phase=phase, deepcopy_model=True)
    if sample_indices is not None:
        inferencer.mutable_dataset_collection.set_subset(
            phase=phase, indices=set(sample_indices)
        )
    res: dict = {}
    if phase == MachineLearningPhase.Training:
        inferencer.replace_model_evaluator(
            lambda model_evaluator: OutputGradientEvaluator(evaluator=model_evaluator)
        )
        sample_loss = inferencer.get_sample_loss(evaluation_mode=EvaluationMode.Test)
        assert sample_loss
        for v in sample_loss.values():
            v.backward(retain_graph=True)
        assert isinstance(inferencer.model_evaluator, OutputGradientEvaluator)
        res |= {"activation_gradients": inferencer.model_evaluator.activation_gradients}

    else:
        inferencer.replace_model_evaluator(
            lambda model_evaluator: OutputModelEvaluator(evaluator=model_evaluator)
        )
        inferencer.inference()
    assert isinstance(inferencer.model_evaluator, OutputFeatureModelEvaluator)
    if sample_indices is not None:
        assert len(inferencer.model_evaluator.output_features) == len(
            set(sample_indices)
        )
    res |= {"output_features": inferencer.model_evaluator.output_features}
    if hasattr(inferencer.model_evaluator, "output_tensors"):
        res |= {"output_tensors": inferencer.model_evaluator.output_tensors}
    return res


def compute_representer_point_values(
    trainer: Trainer,
    test_indices: IndicesType,
    training_indices: OptionalIndicesType = None,
) -> SubsetContribution:
    test_res = __get_output(
        trainer=trainer,
        phase=MachineLearningPhase.Test,
        sample_indices=test_indices,
    )
    test_features = test_res["output_features"]
    test_output_tensors = test_res["output_tensors"]

    training_res = __get_output(
        trainer=trainer,
        phase=MachineLearningPhase.Training,
        sample_indices=training_indices,
    )
    training_features = training_res["output_features"]
    activation_gradients = training_res["activation_gradients"]
    contribution = SubsetContribution()
    contribution.set_tracked_indices(list(training_features.keys()))
    for test_idx, test_feature in test_features.items():
        cls_idx = test_output_tensors[test_idx].argmax().item()
        for training_idx, training_feature in training_features.items():
            product = dot_product(test_feature, training_feature)
            value = math.fabs((activation_gradients[training_idx] * product)[cls_idx])
            contribution.set_sample_contribution(
                tracked_index=training_idx, value=value, test_index=test_idx
            )
    return contribution
