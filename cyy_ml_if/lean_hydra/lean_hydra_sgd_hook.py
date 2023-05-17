from cyy_ml_if.arithmetic_util import (optional_addition,
                                       optional_multiplication)
from cyy_naive_lib.log import get_logger
from cyy_naive_lib.time_counter import TimeCounter

from .lean_hydra_hook import LeanHyDRAHook


class LeanHyDRASGDHook(LeanHyDRAHook):
    __mom_product = None

    def _after_batch(self, batch_size, **kwargs):
        counter = TimeCounter()
        optimizer = self._get_optimizer(**kwargs)

        assert len(optimizer.param_groups) == 1
        momentum = optimizer.param_groups[0]["momentum"]
        lr = optimizer.param_groups[0]["lr"]
        assert lr
        weight_decay = optimizer.param_groups[0]["weight_decay"]
        self.__mom_product = optional_addition(
            optional_multiplication(self.__mom_product, momentum),
            optional_multiplication(self._contributions, weight_decay),
        )

        for idx, dot_product in self._sample_gradient_hook.result_dict.items():
            self.__mom_product[idx] += (
                dot_product * self._training_set_size / batch_size
            )
        self._sample_gradient_hook.reset_result()
        assert id(self._contributions) != id(self.__mom_product)
        self._contributions -= lr * self.__mom_product
        get_logger().debug(
            "batch use time %s ms",
            counter.elapsed_milliseconds(),
        )

    def _after_execute(self, **kwargs):
        assert self._contributions is not None
        self._contributions = -self._contributions / self._training_set_size
        super()._after_execute(**kwargs)
