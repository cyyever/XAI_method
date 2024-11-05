from typing import Any

from cyy_naive_lib.log import get_logger
from cyy_naive_lib.time_counter import TimeCounter
from cyy_torch_toolbox import OptionalTensor

from cyy_torch_xai.arithmetic_util import optional_addition, optional_multiplication

from .lean_hydra_hook import LeanHyDRAHook


class LeanHyDRASGDHook(LeanHyDRAHook):
    __mom_product: OptionalTensor = None

    def _after_batch(self, batch_size, **kwargs) -> None:
        counter = TimeCounter()
        optimizer = self._get_optimizer(**kwargs)

        assert len(optimizer.param_groups) == 1
        momentum = optimizer.param_groups[0]["momentum"]
        lr = optimizer.param_groups[0]["lr"]
        assert lr
        weight_decay = optimizer.param_groups[0]["weight_decay"]
        self.__mom_product = optional_addition(
            optional_multiplication(self.__mom_product, momentum),
            optional_multiplication(self.contribution_tensor, weight_decay),
        )

        assert self.__mom_product is not None

        for idx, dot_product in self._sample_gradient_hook.result_dict.items():
            self.__mom_product[idx] += dot_product * self.training_set_size / batch_size
        self._sample_gradient_hook.reset_result()
        assert id(self._contribution_tensor) != id(self.__mom_product)
        self._contribution_tensor -= lr * self.__mom_product
        get_logger().debug(
            "batch use time %s ms",
            counter.elapsed_milliseconds(),
        )

    def _after_execute(self, **kwargs: Any) -> None:
        self._contribution_tensor = -self.contribution_tensor / self.training_set_size
        super()._after_execute(**kwargs)
