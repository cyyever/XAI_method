from typing import Any

import torch
from cyy_torch_algorithm.computation.sample_gradient.sample_gradient_hook import \
    get_sample_gradients
from cyy_torch_toolbox import IterationUnit, MachineLearningPhase
from cyy_torch_toolbox.hook import Hook
from cyy_torch_toolbox.typing import OptionalIndicesType, SampleGradients

from ..typing import SampleContributions


class TracInBaseHook(Hook):
    def __init__(
        self,
        test_sample_indices: OptionalIndicesType = None,
        check_point_gap: tuple[int, IterationUnit] | None = None,
    ) -> None:
        super().__init__(stripable=True)

        self.__check_point_gap = check_point_gap
        self.__test_sample_indices: OptionalIndicesType = test_sample_indices
        self.__test_grad_dict: SampleGradients = {}
        self._influence_values: dict[int, SampleContributions] = {}
        self.__batch_num: int = 0

    @property
    def test_grad_dict(self) -> SampleGradients:
        return self.__test_grad_dict

    @property
    def influence_values(self) -> dict[int, SampleContributions]:
        return self._influence_values

    def _before_execute(self, **kwargs: Any) -> None:
        self._influence_values = {}

    def __compute_test_sample_gradients(self, executor, **kwargs: Any) -> None:
        if self.__batch_num != 0 and self.__check_point_gap is not None:
            gap, unit = self.__check_point_gap
            match unit:
                case IterationUnit.Batch:
                    if self.__batch_num % gap != 0:
                        return
                case IterationUnit.Epoch:
                    if kwargs["epoch"] % gap != 0:
                        return
                case _:
                    raise RuntimeError(unit)
        inferencer = executor.get_inferencer(phase=MachineLearningPhase.Test)
        if self.__test_sample_indices is None:
            self.__test_grad_dict[-1] = inferencer.get_gradient()
        else:
            self.__test_grad_dict.update(
                get_sample_gradients(
                    inferencer=inferencer,
                    computed_indices=self.__test_sample_indices,
                )
            )

    @torch.no_grad()
    def _before_batch(self, **kwargs) -> None:
        self.__compute_test_sample_gradients(**kwargs)
        self.__batch_num += 1
