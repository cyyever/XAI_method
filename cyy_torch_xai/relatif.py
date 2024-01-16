from typing import Iterable

import torch
import torch.linalg
from cyy_naive_lib.algorithm.mapping_op import get_mapping_values_by_key_order
from cyy_torch_algorithm.computation.sample_gradient import \
    get_sample_gradients
from cyy_torch_toolbox import MachineLearningPhase, Trainer
from cyy_torch_toolbox.tensor import cat_tensor_dict, dot_product
from cyy_torch_toolbox.typing import TensorDict

from .inverse_hessian_vector_product import \
    stochastic_inverse_hessian_vector_product
from .util import get_test_gradient


def compute_relatif(
    trainer: Trainer,
    training_indices: Iterable[int] | None,
    test_gradient: TensorDict | None = None,
) -> dict[int, float]:
    if test_gradient is None:
        test_gradient = get_test_gradient(trainer=trainer)

    inferencer = trainer.get_inferencer(
        phase=MachineLearningPhase.Training, deepcopy_model=False
    )
    training_sample_gradients = get_sample_gradients(
        inferencer=inferencer, computed_indices=training_indices
    )
    products = stochastic_inverse_hessian_vector_product(
        inferencer,
        vectors=list(get_mapping_values_by_key_order(training_sample_gradients)),
    )
    return [
        dot_product(product, test_gradient)
        / torch.linalg.vector_norm(cat_tensor_dict(product)).item()
        for product in products
    ]


# from .influence_function import get_default_inverse_hvp_arguments
# from .inverse_hessian_vector_product import \
#     stochastic_inverse_hessian_vector_product
# from .util import compute_perturbation_gradient_difference


# def compute_perturbation_relatif(
#     trainer: Trainer,
#     perturbation_idx_fun: Callable,
#     perturbation_fun: Callable,
#     test_gradient: torch.Tensor | None = None,
#     inverse_hvp_arguments: None | dict = None,
#     grad_diff=None,
# ) -> dict:
#     if test_gradient is None:
#         inferencer = trainer.get_inferencer(
#             phase=MachineLearningPhase.Test, deepcopy_model=True
#         )
#         test_gradient = inferencer.get_gradient()
#     test_gradient = test_gradient.cpu()

#     if grad_diff is None:
#         grad_diff = compute_perturbation_gradient_difference(
#             trainer=trainer,
#             perturbation_idx_fun=perturbation_idx_fun,
#             perturbation_fun=perturbation_fun,
#         )

#     if inverse_hvp_arguments is None:
#         inverse_hvp_arguments = get_default_inverse_hvp_arguments()
#         inverse_hvp_arguments["repeated_num"] = 1

#     res: dict = {}
#     accumulated_indices = []
#     accumulated_vectors = []
#     inferencer = trainer.get_inferencer(
#         phase=MachineLearningPhase.Training, deepcopy_model=True
#     )
#     batch_size = 32
#     for perturbation_idx, v in grad_diff.items():
#         v_norm = v.norm()
#         # normalize to 1 makes convergence easier
#         if v_norm.item() > 1:
#             v = v / v_norm
#         get_logger().error("v norm is %s", v_norm)
#         accumulated_indices.append(perturbation_idx)
#         accumulated_vectors.append(v)
#         if len(accumulated_indices) != batch_size:
#             continue
#         products = stochastic_inverse_hessian_vector_product(
#             inferencer, vectors=accumulated_vectors, **inverse_hvp_arguments
#         )
#         for idx, product in zip(accumulated_indices, products):
#             res[idx] = (-test_gradient.dot(product) / product.norm()).item()
#         accumulated_indices = []
#         accumulated_vectors = []
#     if accumulated_indices:
#         products = stochastic_inverse_hessian_vector_product(
#             inferencer, vectors=accumulated_vectors, **inverse_hvp_arguments
#         )
#         for idx, product in zip(accumulated_indices, products):
#             res[idx] = (
#                 -test_gradient.dot(product) / torch.linalg.vector_norm(product)
#             ).item()
#     assert len(res) == len(grad_diff)
#     return res
