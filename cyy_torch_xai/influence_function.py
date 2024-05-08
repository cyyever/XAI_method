from typing import Iterable

from cyy_naive_lib.algorithm.mapping_op import get_mapping_values_by_key_order
from cyy_torch_algorithm.computation.sample_gradient import (
    get_sample_gradients, get_sample_gvps, get_self_gvps)
from cyy_torch_toolbox import MachineLearningPhase, Trainer
from cyy_torch_toolbox.typing import TensorDict

from .inverse_hessian_vector_product import \
    stochastic_inverse_hessian_vector_product
from .util import get_test_gradient


def compute_influence_function_values(
    trainer: Trainer,
    computed_indices: Iterable[int] | None = None,
    test_gradient: TensorDict | None = None,
) -> dict[int, float]:
    if test_gradient is None:
        test_gradient = get_test_gradient(trainer=trainer)

    inferencer = trainer.get_inferencer(
        phase=MachineLearningPhase.Training, deepcopy_model=False
    )
    product = stochastic_inverse_hessian_vector_product(
        inferencer, vectors=[test_gradient]
    )[0]

    products = get_sample_gvps(
        inferencer=inferencer, vector=product, computed_indices=computed_indices
    )
    return {idx: product / trainer.dataset_size for idx, product in products.items()}


def compute_self_influence_function_values(
    trainer: Trainer,
    computed_indices: set[int],
) -> dict:
    inferencer = trainer.get_inferencer(
        phase=MachineLearningPhase.Training, deepcopy_model=False
    )
    test_gradients = get_sample_gradients(
        inferencer=inferencer, computed_indices=computed_indices
    )

    products = stochastic_inverse_hessian_vector_product(
        inferencer, vectors=list(get_mapping_values_by_key_order(test_gradients))
    )

    gvps = get_self_gvps(
        inferencer=inferencer,
        vectors=dict(zip(sorted(computed_indices), products)),
    )

    return {idx: product / trainer.dataset_size for idx, product in gvps.items()}
