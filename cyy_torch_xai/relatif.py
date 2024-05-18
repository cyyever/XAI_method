import torch
import torch.linalg
from cyy_naive_lib.algorithm.mapping_op import get_mapping_values_by_key_order
from cyy_torch_algorithm.computation.sample_gradient import \
    get_sample_gradients
from cyy_torch_toolbox import MachineLearningPhase, Trainer
from cyy_torch_toolbox.tensor import cat_tensor_dict, dot_product
from cyy_torch_toolbox.typing import ModelGradient, OptionalIndicesType

from .inverse_hessian_vector_product import \
    stochastic_inverse_hessian_vector_product
from .typing import SampleContributionDict
from .util import get_test_gradient


def compute_relatif_values(
    trainer: Trainer,
    computed_indices: OptionalIndicesType = None,
    test_gradient: ModelGradient | None = None,
) -> SampleContributionDict:
    if test_gradient is None:
        test_gradient = get_test_gradient(trainer=trainer)

    inferencer = trainer.get_inferencer(
        phase=MachineLearningPhase.Training, deepcopy_model=False
    )
    training_sample_gradients = get_sample_gradients(
        inferencer=inferencer, computed_indices=computed_indices
    )
    products = stochastic_inverse_hessian_vector_product(
        inferencer,
        vectors=list(get_mapping_values_by_key_order(training_sample_gradients)),
    )
    return {
        idx: dot_product(product, test_gradient)
        / torch.linalg.vector_norm(cat_tensor_dict(product)).item()
        for idx, product in zip(sorted(training_sample_gradients.keys()), products)
    }
