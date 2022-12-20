import torch
from cyy_naive_lib.log import get_logger
from cyy_naive_lib.time_counter import TimeCounter

from .lean_hydra_hook import LeanHyDRAHook


class LeanHyDRASGDHook(LeanHyDRAHook):
    __mom_product = None
    __momentum = None
    __lr = None
    __weight_decay = None

    def _before_batch(self, **kwargs):
        optimizer = self._get_optimizer(**kwargs)

        assert len(optimizer.param_groups) == 1
        self.__momentum = optimizer.param_groups[0]["momentum"]
        self.__lr = optimizer.param_groups[0]["lr"]
        self.__weight_decay = optimizer.param_groups[0]["weight_decay"]

    def _after_optimizer_step(self, batch_size, step_skipped, **kwargs):
        if step_skipped:
            self.sample_gradient_hook.reset_result()
            return

        counter = TimeCounter()
        if self.__mom_product is None:
            self.__mom_product = torch.zeros(self._training_set_size).to(
                self._contributions.device, non_blocking=True
            )
        self.__mom_product = (
            self.__mom_product * self.__momentum
            + self.__weight_decay * self._contributions
        )

        for idx, dot_product in self.sample_gradient_hook.result_dict.items():
            self.__mom_product[idx] += (
                dot_product * self._training_set_size / batch_size
            )
        self.sample_gradient_hook.reset_result()
        self._contributions -= self.__lr * self.__mom_product
        get_logger().debug(
            "batch use time %s ms",
            counter.elapsed_milliseconds(),
        )

    def _after_execute(self, **kwargs):
        self._contributions = -self._contributions / self._training_set_size
        super()._after_execute(**kwargs)
