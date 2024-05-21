from typing import Any

import torch
from cyy_torch_algorithm.computation.sample_gradient.sample_gradient_hook import \
    get_sample_gradients
from cyy_torch_toolbox import (IterationUnit, MachineLearningPhase,
                               OptionalIndicesType, SampleGradients)

from ..base_hook import SampleGradientXAIHook


class TracInBaseHook(SampleGradientXAIHook):
    def __init__(
        self,
        test_sample_indices: OptionalIndicesType = None,
        check_point_gap: tuple[int, IterationUnit] | None = None,
    ) -> None:
        super().__init__()
        self.__check_point_gap = check_point_gap
        self.__test_sample_indices: OptionalIndicesType = test_sample_indices
        self.__test_gradients: SampleGradients = {}
        self.__batch_num: int = 0

    @property
    def test_gradients(self) -> SampleGradients:
        return self.__test_gradients

    def __compute_test_sample_gradients(self, executor, **kwargs: Any) -> None:
        if self.__test_gradients:
            if self.__check_point_gap is not None:
                gap, unit = self.__check_point_gap
                match unit:
                    case IterationUnit.Batch:
                        if self.__batch_num % gap != 0:
                            return
                    case IterationUnit.Epoch:
                        epoch = kwargs["epoch"]
                        if epoch % gap != 0:
                            return
                    case _:
                        raise RuntimeError(unit)
        inferencer = executor.get_inferencer(phase=MachineLearningPhase.Test)
        if self.__test_sample_indices is None:
            self.__test_gradients[-1] = inferencer.get_gradient()
        else:
            self.__test_gradients.update(
                get_sample_gradients(
                    inferencer=inferencer,
                    computed_indices=self.__test_sample_indices,
                )
            )

    @torch.no_grad()
    def _before_batch(self, **kwargs: Any) -> None:
        self.__compute_test_sample_gradients(**kwargs)
        self.__batch_num += 1
