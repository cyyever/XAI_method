import json
import os
from typing import Any

import torch
from cyy_torch_toolbox import ModelGradient, OptionalTensor, tensor_to
from cyy_torch_toolbox.tensor import dot_product

from ..hydra.base_hook import BaseHook
from ..typing import SampleContributions


class LeanHyDRAHook(BaseHook):
    def __init__(self, test_gradient: ModelGradient, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.__test_gradient: ModelGradient = tensor_to(test_gradient, device="cpu")
        self._sample_gradient_hook.set_result_transform(self._dot_product)
        if self._batch_hvp_hook is not None:
            self._batch_hvp_hook.set_vectors([self.__test_gradient])
        self._contribution_tensor: OptionalTensor = None

    @property
    def test_gradient(self):
        return self.__test_gradient

    @property
    def contribution_tensor(self) -> torch.Tensor:
        assert self._contribution_tensor is not None
        return self._contribution_tensor

    @property
    def contribution_dict(self) -> SampleContributions:
        return {
            idx: self.contribution_tensor[idx].item() for idx in self.computed_indices
        }

    def _before_execute(self, **kwargs: Any) -> None:
        super()._before_execute(**kwargs)
        self._contribution_tensor = torch.zeros(self.training_set_size)

    def _dot_product(self, result, **kwargs: Any) -> float:
        return dot_product(self.test_gradient, result)

    def _after_execute(self, **kwargs: Any) -> None:
        assert self.contribution_tensor.shape[0] == self.training_set_size
        save_dir = "."
        if "executor" in kwargs:
            trainer = kwargs["executor"]
            save_dir = os.path.join(trainer.save_dir, "lean_HyDRA")
            os.makedirs(save_dir, exist_ok=True)

        with open(
            os.path.join(save_dir, "lean_hydra_contribution.json"),
            mode="w",
            encoding="utf-8",
        ) as f:
            json.dump(self.contribution_dict, f)
        super()._after_execute(**kwargs)
