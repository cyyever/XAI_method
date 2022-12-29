from cyy_ml_if.util import (optional_addition, optional_multiplication,
                            optional_subtraction)
# from cyy_naive_lib.log import get_logger
# from cyy_naive_lib.time_counter import TimeCounter
from cyy_torch_toolbox.tensor import cat_tensors_to_vector

from .lean_hydra_hook import LeanHyDRAHook


class LeanHyDRAAdamHook(LeanHyDRAHook):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__gradient_product = None
        self.__first_average_product = None
        self.__second_average_product = None
        self.__first_average = None
        self.__second_average = None

    def _after_optimizer_step(self, step_skipped, batch_size, **kwargs):
        if step_skipped:
            return

        optimizer = self._get_optimizer(**kwargs)
        step = list(optimizer.state.values())[0]["step"]
        assert len(optimizer.param_groups) == 1
        lr = optimizer.param_groups[0]["lr"]
        beta1, beta2 = optimizer.param_groups[0]["betas"]
        weight_decay = optimizer.param_groups[0]["weight_decay"]
        parameter_seq = tuple(
            self._get_model_util(**kwargs).get_parameter_seq(detach=False)
        )

        self.__first_average = cat_tensors_to_vector(
            (optimizer.state[p]["exp_avg"] for p in parameter_seq)
        )
        self.__second_average = cat_tensors_to_vector(
            (optimizer.state[p]["exp_avg_sq"] for p in parameter_seq)
        )
        # TODO fix mean
        corrected_first_average = self.__first_average / (1 - (beta1**step))
        corrected_second_average = self.__second_average / (1 - (beta2**step))
        eps = optimizer.param_groups[0]["eps"]

        self.__gradient_product = optional_multiplication(
            self._contributions, weight_decay
        )
        for idx, dot_product in self._sample_gradient_hook.result_dict.items():
            self.__gradient_product[idx] += (
                dot_product * self._training_set_size / batch_size
            )
        self._sample_gradient_hook.reset_result()

        self.__first_average_product = optional_addition(
            optional_multiplication(self.__first_average_product, beta1),
            optional_multiplication(self.__gradient_product, (1 - beta1)),
        ) / (1 - (beta1**step))

        self.__second_average_product = optional_addition(
            optional_multiplication(self.__second_average_product, beta2),
            optional_multiplication(self.__gradient_product, 2 * (1 - beta2)),
        ) / (1 - (beta2**step))
        corrected_second_average_sqrt = corrected_second_average.sqrt().mean().item()

        self._contributions = optional_subtraction(
            self._contributions,
            optional_multiplication(
                optional_addition(
                    optional_multiplication(
                        self.__first_average_product,
                        corrected_second_average_sqrt + eps,
                    ),
                    optional_multiplication(
                        self.__second_average_product,
                        corrected_first_average.mean().item()
                        / (corrected_second_average_sqrt * 2),
                    ),
                ),
                lr / ((corrected_second_average_sqrt + eps) ** 2),
            ),
        )
