import json
import os

from cyy_torch_algorithm.computation.sample_gradient.sample_gradient_hook import \
    sample_dot_product
from cyy_torch_toolbox.tensor import tensor_to

from ..hydra.base_hook import BaseHook


class LeanHyDRAHook(BaseHook):
    def __init__(self, test_gradient, **kwargs) -> None:
        super().__init__(**kwargs)
        self.__test_gradient = tensor_to(test_gradient, device="cpu")
        self._sample_gradient_hook.set_result_transform(self._gradient_dot_product)
        if self._batch_hvp_hook is not None:
            self._batch_hvp_hook.set_vectors([self.__test_gradient])

    def _gradient_dot_product(self, result, **kwargs):
        return sample_dot_product(
            self.__test_gradient, tensor_to(result, non_blocking=True, device="cpu")
        )

    def _after_execute(self, **kwargs) -> None:
        assert self._contributions is not None
        assert self._contributions.shape[0] == self._training_set_size
        save_dir = "."
        if "executor" in kwargs:
            trainer = kwargs["executor"]
            save_dir = os.path.join(trainer.save_dir, "lean_HyDRA")
            os.makedirs(save_dir, exist_ok=True)

        with open(
            os.path.join(save_dir, "lean_hydra_contribution.json"),
            mode="wt",
            encoding="utf-8",
        ) as f:
            contributions = self._contributions.cpu().tolist()
            json.dump({idx: contributions[idx] for idx in self._computed_indices}, f)
        super()._after_execute(**kwargs)
