from typing import Any

import torch
import torch.nn
import torch.utils.hooks
from cyy_naive_lib.decorator import Decorator
from cyy_torch_toolbox import ModelEvaluator, SampleTensors


class OutputFeatureModelEvaluator(Decorator):
    def __init__(self, evaluator: ModelEvaluator) -> None:
        super().__init__(evaluator)
        self._sample_indices: list = []
        self.__output_features: SampleTensors = {}
        self.last_module: torch.nn.Module = self.model_util.get_last_underlying_module()
        assert isinstance(self.last_module, torch.nn.Linear)
        self.last_module.register_forward_pre_hook(
            hook=self.__feature_hook_impl, with_kwargs=True
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self._sample_indices = kwargs["sample_indices"].tolist()
        return self._decorator_object.__call__(*args, **kwargs)

    @property
    def output_features(self) -> SampleTensors:
        return self.__output_features

    def __feature_hook_impl(self, module, *args: Any, **kwargs: Any) -> Any:
        input_tensor: torch.Tensor = args[0][0]
        assert input_tensor.shape[0] == len(self._sample_indices)
        self.__output_features |= dict(
            zip(self._sample_indices, input_tensor.clone(), strict=False)
        )
        return None


class OutputGradientEvaluator(OutputFeatureModelEvaluator):
    def __init__(self, evaluator: ModelEvaluator) -> None:
        super().__init__(evaluator=evaluator)
        self.last_module.register_forward_hook(
            hook=self.__output_hook_impl, with_kwargs=True
        )
        self.__accumulated_sample_indices: list[list] = []
        self.__layer_output_tensors: list[torch.Tensor] = []

    @property
    def activation_gradients(self) -> SampleTensors:
        res: SampleTensors = {}
        for a, b in zip(
            self.__accumulated_sample_indices, self.__layer_output_tensors, strict=False
        ):
            assert b.grad is not None
            assert len(a) == b.grad.shape[0]
            res |= dict(zip(a, b.grad, strict=False))
        return res

    def __output_hook_impl(self, module, *args: Any, **kwargs: Any) -> Any:
        output_tensor: torch.Tensor = args[0][0]
        assert output_tensor.shape[0] == len(self._sample_indices)
        assert output_tensor.grad is None
        output_tensor.requires_grad_()
        output_tensor.retain_grad()
        self.__accumulated_sample_indices.append(self._sample_indices)
        self.__layer_output_tensors.append(output_tensor)
        return None


class OutputModelEvaluator(OutputFeatureModelEvaluator):
    def __init__(self, evaluator: ModelEvaluator) -> None:
        super().__init__(evaluator=evaluator)
        self.__output_tensors: SampleTensors = {}
        last_module = list(reversed(list(self.model_util.get_modules())))[0][1]
        last_module.register_forward_hook(
            hook=self.__output_hook_impl, with_kwargs=True
        )

    @property
    def output_tensors(self) -> SampleTensors:
        return self.__output_tensors

    def __output_hook_impl(self, module, *args: Any, **kwargs: Any) -> Any:
        input_tensor: torch.Tensor = args[0][0]
        assert input_tensor.shape[0] == len(self._sample_indices)
        self.__output_tensors |= dict(
            zip(self._sample_indices, input_tensor.clone(), strict=False)
        )
        return None
