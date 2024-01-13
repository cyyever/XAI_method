from cyy_torch_algorithm.computation.sample_gradient import get_sample_gvp_dict
from cyy_torch_toolbox import MachineLearningPhase, Trainer
from cyy_torch_toolbox.typing import TensorDict

from .inverse_hessian_vector_product import \
    stochastic_inverse_hessian_vector_product
from .util import get_test_gradient


def get_default_inverse_hvp_arguments() -> dict:
    return {"dampling_term": 0.01, "scale": 100000, "epsilon": 0.03, "repeated_num": 3}


def compute_influence_function(
    trainer: Trainer,
    training_indices: set | None,
    inverse_hvp_arguments: None | dict = None,
    test_gradient: TensorDict | None = None,
) -> dict:
    if test_gradient is None:
        test_gradient = get_test_gradient(trainer=trainer)

    inferencer = trainer.get_inferencer(
        phase=MachineLearningPhase.Training, deepcopy_model=True
    )
    if inverse_hvp_arguments is None:
        inverse_hvp_arguments = get_default_inverse_hvp_arguments()
    product = stochastic_inverse_hessian_vector_product(
        inferencer, vectors=[test_gradient], **inverse_hvp_arguments
    )[0]

    res = get_sample_gvp_dict(
        inferencer=inferencer, vector=product, computed_indices=training_indices
    )
    return {k: v / trainer.dataset_size for k, v in res.items()}


# def compute_perturbation_influence_function(
#     trainer: Trainer,
#     perturbation_idx_fun: Callable,
#     perturbation_fun: Callable,
#     test_gradient: dict | None = None,
#     inverse_hvp_arguments: None | dict = None,
#     grad_diff=None,
# ) -> dict:
#     if test_gradient is None:
#         test_gradient = get_test_gradient(trainer=trainer)

#     inferencer = trainer.get_inferencer(
#         phase=MachineLearningPhase.Training, deepcopy_model=True
#     )
#     if inverse_hvp_arguments is None:
#         inverse_hvp_arguments = get_default_inverse_hvp_arguments()

#     product = (
#         -stochastic_inverse_hessian_vector_product(
#             inferencer, vectors=[test_gradient], **inverse_hvp_arguments
#         )
#         / trainer.dataset_size
#     )[0].cpu()
#     if grad_diff is not None:
#         res = {}
#         for perturbation_idx, v in grad_diff.items():
#             res[perturbation_idx] = v.cpu().dot(product).item()
#         return res

#     return compute_perturbation_gradient_difference(
#         trainer=trainer,
#         perturbation_idx_fun=perturbation_idx_fun,
#         perturbation_fun=perturbation_fun,
#         result_transform=functools.partial(dot_product, b=product),
#     )

# def compute_self_influence_function(
#     trainer: Trainer,
#     computed_indices: set,
#     inverse_hvp_arguments: None | dict = None,
# ) -> dict:
#     inferencer = trainer.get_inferencer(
#         phase=MachineLearningPhase.Training, deepcopy_model=True
#     )
#     test_gradients: dict = get_sample_gradient_dict(
#         inferencer=inferencer, computed_indices=computed_indices
#     )

#     if inverse_hvp_arguments is None:
#         inverse_hvp_arguments = get_default_inverse_hvp_arguments()
#     products = (
#         stochastic_inverse_hessian_vector_product(
#             inferencer,
#             vectors=list(get_mapping_values_by_key_order(test_gradients)),
#             **inverse_hvp_arguments,
#         )
#         / trainer.dataset_size
#     ).cpu()

#     return get_self_gvp_dict(
#         inferencer=inferencer,
#         vectors=dict(zip(sorted(computed_indices), products)),
#     )
