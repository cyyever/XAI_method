import torch

from cyy_naive_lib.log import log_error
from cyy_torch_toolbox import (IndicesType, MachineLearningPhase,
                               OptionalIndicesType, Trainer, EvaluationMode)

from cyy_torch_toolbox.tensor import dot_product
from .evaluator import OutputFeatureModelEvaluator, OutputModelEvaluator, OutputGradientEvaluator
from ..contribution import SubsetContribution


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
        res |= {"layer_gradients": inferencer.model_evaluator.layer_output_gradients}

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
    log_error("aaa %s", len(test_output_tensors))
    log_error("vvv %s", len(training_features))
    contribution = SubsetContribution()
    contribution.set_tracked_indices(list(training_features.keys()))
    for test_idx, test_feature in test_features.items():
        for training_idx, training_feature in training_features.items():
            product = dot_product(test_feature, training_feature)
            contribution.set_sample_contribution(tracked_index=training_idx, value=product, test_index=test_idx)
    return contribution
