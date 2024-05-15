from typing import Any

import torch
import torch.nn
import torch.utils.hooks
from cyy_torch_toolbox import ModelEvaluator


class OutputFeatureModelEvaluator:
    def __init__(self, evaluator: ModelEvaluator) -> None:
        self.evaluator: ModelEvaluator = evaluator
        self.__sample_indices: list = []
        self.__output_features: dict[int, torch.Tensor] = {}
        last_module = self.evaluator.model_util.get_last_underlying_module()
        assert isinstance(last_module, torch.nn.Linear)
        last_module.register_forward_pre_hook(
            hook=self.__feature_hook_impl, with_kwargs=True
        )

    @property
    def output_features(self) -> dict:
        return self.__output_features

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.__sample_indices = kwargs["sample_indices"].tolist()
        return self.evaluator.__call__(*args, **kwargs)

    def __feature_hook_impl(self, module, *args, **kwargs) -> Any:
        input_tensor: torch.Tensor = args[0][0]
        assert input_tensor.shape[0] == len(self.__sample_indices)
        self.__output_features |= dict(zip(self.__sample_indices, input_tensor.clone()))
        return None

    def __getattr__(self, name: str) -> Any:
        if name == "evaluator":
            raise AttributeError()
        return getattr(self.evaluator, name)
