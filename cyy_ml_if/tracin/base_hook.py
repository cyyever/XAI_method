import torch
from cyy_torch_algorithm.computation.sample_gradient.sample_gradient_hook import \
    get_sample_gradient_dict
from cyy_torch_algorithm.data_structure.synced_tensor_dict import \
    SyncedTensorDict
from cyy_torch_toolbox.hook import Hook
from cyy_torch_toolbox.ml_type import MachineLearningPhase


class TracInBaseHook(Hook):
    def __init__(
        self, test_sample_indices: set | None = None, check_point_gap: int | None = None
    ):
        super().__init__(stripable=True)

        self.__check_point_gap = check_point_gap
        self.__test_sample_indices = test_sample_indices
        self.__test_grad_dict = SyncedTensorDict.create()
        self._influence_values: dict = {}

    @property
    def test_grad_dict(self) -> dict:
        return self.__test_grad_dict

    @property
    def influence_values(self) -> dict:
        return self._influence_values

    @torch.no_grad()
    def _before_execute(self, **kwargs):
        self._influence_values = {}

    def __compute_test_sample_gradients(self, model_executor, batch_index):
        if batch_index != 0 and (
            self.__check_point_gap is not None
            and batch_index % self.__check_point_gap != 0
        ):
            return
        inferencer = model_executor.get_inferencer(phase=MachineLearningPhase.Test)
        inferencer.disable_logger()
        inferencer.disable_performance_metric_logger()
        if self.__test_sample_indices is None:
            self.__test_grad_dict[-1] = inferencer.get_gradient()
        else:

            def collect_result(res_dict):
                for k, v in res_dict.items():
                    self.__test_grad_dict[k] = v

            get_sample_gradient_dict(
                inferencer=inferencer,
                computed_indices=self.__test_sample_indices,
                result_collection_fun=collect_result,
            )

    @torch.no_grad()
    def _before_batch(self, model_executor, batch_index, **kwargs):
        self.__compute_test_sample_gradients(
            model_executor=model_executor, batch_index=batch_index
        )
