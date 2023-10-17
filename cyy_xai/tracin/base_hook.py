import torch
from cyy_torch_algorithm.computation.sample_gradient.sample_gradient_hook import \
    get_sample_gradient_dict
from cyy_torch_toolbox.hook import Hook
from cyy_torch_toolbox.ml_type import MachineLearningPhase


class TracInBaseHook(Hook):
    def __init__(
        self, test_sample_indices: set | None = None, check_point_gap: int | None = None
    ):
        super().__init__(stripable=True)

        self.__check_point_gap = check_point_gap
        self.__test_sample_indices = test_sample_indices
        self.__test_grad_dict: dict = {}
        self._influence_values: dict = {}
        self.__batch_num = 0

    @property
    def test_grad_dict(self) -> dict:
        return self.__test_grad_dict

    @property
    def influence_values(self) -> dict:
        return self._influence_values

    @torch.no_grad()
    def _before_execute(self, **kwargs):
        self._influence_values = {}

    def __compute_test_sample_gradients(self, executor):
        if (
            self.__batch_num != 0
            and self.__check_point_gap is not None
            and self.__batch_num % self.__check_point_gap != 0
        ):
            return
        inferencer = executor.get_inferencer(phase=MachineLearningPhase.Test)
        inferencer.disable_hook("logger")
        inferencer.disable_hook("performance_metric")
        if self.__test_sample_indices is None:
            self.__test_grad_dict[-1] = inferencer.get_gradient()
        else:

            def collect_result(res_dict):
                self.__test_grad_dict |= res_dict

            get_sample_gradient_dict(
                inferencer=inferencer,
                computed_indices=self.__test_sample_indices,
                result_collection_fun=collect_result,
            )

    @torch.no_grad()
    def _before_batch(self, executor, batch_index, **kwargs):
        self.__compute_test_sample_gradients(executor=executor)
        self.__batch_num += 1
