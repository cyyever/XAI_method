import functools
from typing import Callable

import torch
from cyy_naive_lib.algorithm.mapping_op import get_mapping_values_by_key_order
from cyy_torch_algorithm.computation.sample_gradient.sample_gradient_hook import (
    dot_product, get_sample_gradient_dict, get_sample_gvp_dict,
    get_self_gvp_dict)
from cyy_torch_toolbox.ml_type import MachineLearningPhase
from cyy_torch_toolbox.trainer import Trainer

from cyy_torch_xai.inverse_hessian_vector_product import \
    stochastic_inverse_hessian_vector_product
from cyy_torch_xai.util import compute_perturbation_gradient_difference


def get_default_inverse_hvp_arguments() -> dict:
    return {"dampling_term": 0.01, "scale": 100000, "epsilon": 0.03, "repeated_num": 3}


def compute_self_influence_function(
    trainer: Trainer,
    computed_indices: set,
    inverse_hvp_arguments: None | dict = None,
) -> dict:

    inferencer = trainer.get_inferencer(
        phase=MachineLearningPhase.Training, deepcopy_model=True
    )
    test_gradients: dict = get_sample_gradient_dict(
        inferencer=inferencer, computed_indices=computed_indices
    )

    if inverse_hvp_arguments is None:
        inverse_hvp_arguments = get_default_inverse_hvp_arguments()
    products = (
        stochastic_inverse_hessian_vector_product(
            inferencer,
            vectors=list(get_mapping_values_by_key_order(test_gradients)),
            **inverse_hvp_arguments
        )
        / trainer.dataset_size
    ).cpu()

    return get_self_gvp_dict(
        inferencer=inferencer,
        vectors=dict(zip(sorted(computed_indices), products)),
    )


def compute_influence_function(
    trainer: Trainer,
    computed_indices: set | None,
    inverse_hvp_arguments: None | dict = None,
) -> dict:
    inferencer = trainer.get_inferencer(
        phase=MachineLearningPhase.Test, deepcopy_model=True
    )
    test_gradient = inferencer.get_gradient()
    del inferencer

    inferencer = trainer.get_inferencer(
        phase=MachineLearningPhase.Training, deepcopy_model=True
    )
    if inverse_hvp_arguments is None:
        inverse_hvp_arguments = get_default_inverse_hvp_arguments()
    product = (
        stochastic_inverse_hessian_vector_product(
            inferencer, vectors=[test_gradient], **inverse_hvp_arguments
        )
        / trainer.dataset_size
    )[0].cpu()

    return get_sample_gvp_dict(
        inferencer=inferencer, vector=product, computed_indices=computed_indices
    )


def compute_perturbation_influence_function(
    trainer: Trainer,
    perturbation_idx_fun: Callable,
    perturbation_fun: Callable,
    test_gradient: torch.Tensor | None = None,
    inverse_hvp_arguments: None | dict = None,
    grad_diff=None,
) -> dict:
    if test_gradient is None:
        inferencer = trainer.get_inferencer(
            phase=MachineLearningPhase.Test, deepcopy_model=True
        )
        test_gradient = inferencer.get_gradient()

    inferencer = trainer.get_inferencer(
        phase=MachineLearningPhase.Training, deepcopy_model=True
    )
    if inverse_hvp_arguments is None:
        inverse_hvp_arguments = get_default_inverse_hvp_arguments()

    trainer.offload_from_gpu()

    product = (
        -stochastic_inverse_hessian_vector_product(
            inferencer, vectors=[test_gradient], **inverse_hvp_arguments
        )
        / trainer.dataset_size
    )[0].cpu()
    if grad_diff is not None:
        res = {}
        for perturbation_idx, v in grad_diff.items():
            res[perturbation_idx] = v.cpu().dot(product).item()
        return res

    return compute_perturbation_gradient_difference(
        trainer=trainer,
        perturbation_idx_fun=perturbation_idx_fun,
        perturbation_fun=perturbation_fun,
        result_transform=functools.partial(dot_product, vector=product),
    )
