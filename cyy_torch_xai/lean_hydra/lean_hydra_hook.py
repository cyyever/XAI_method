import json
import os
from typing import Any

import torch
from cyy_torch_algorithm.computation import dot_product
from cyy_torch_toolbox import ModelGradient, OptionalTensor, tensor_to

from ..hydra.base_hook import BaseHook


class LeanHyDRAHook(BaseHook):
    def __init__(self, test_gradient: ModelGradient, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.__test_gradient = tensor_to(test_gradient, device="cpu")
        self._sample_gradient_hook.set_result_transform(self._dot_product)
        if self._batch_hvp_hook is not None:
            self._batch_hvp_hook.set_vectors([self.__test_gradient])
        self._contributions: OptionalTensor = None

    @property
    def contributions(self) -> torch.Tensor:
        assert self._contributions is not None
        return self._contributions

    def _before_execute(self, **kwargs: Any) -> None:
        super()._before_execute(**kwargs)
        self._contributions = torch.zeros(self.training_set_size)

    def _dot_product(self, result, **kwargs) -> float:
        return dot_product(self.__test_gradient, result)

    def _after_execute(self, **kwargs) -> None:
        assert self.contributions.shape[0] == self.training_set_size
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
            contributions = self.contributions.cpu().tolist()
            json.dump({idx: contributions[idx] for idx in self.computed_indices}, f)
        super()._after_execute(**kwargs)
