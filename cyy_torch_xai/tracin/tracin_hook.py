import json
import os

import torch
from cyy_torch_xai.tracin.base_hook import TracInBaseHook
from cyy_naive_lib.log import get_logger
from cyy_torch_algorithm.computation.sample_gradient.sample_gradient_hook import (
    SampleGradientHook, dot_product)
from cyy_torch_toolbox.tensor import tensor_to


class TracInHook(TracInBaseHook):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sample_grad_hook: SampleGradientHook = SampleGradientHook()
        self.__tracked_indices: None | set = None

    def set_tracked_indices(self, tracked_indices: set) -> None:
        self.__tracked_indices = set(tracked_indices)
        get_logger().info("track %s indices", len(self.__tracked_indices))

    @torch.no_grad()
    def _before_execute(self, executor, **kwargs):
        super()._before_execute(executor=executor, **kwargs)
        if self.__tracked_indices is not None:
            self._sample_grad_hook.set_computed_indices(self.__tracked_indices)

    def _after_batch(self, executor, batch_size, **kwargs):
        trainer = executor
        optimizer = trainer.get_optimizer()
        assert len(optimizer.param_groups) == 1
        if not isinstance(optimizer, torch.optim.SGD):
            raise RuntimeError("optimizer is not SGD")
        lr = optimizer.param_groups[0]["lr"]

        assert self.test_grad_dict
        for k, test_grad in self.test_grad_dict.items():
            if k not in self._influence_values:
                self._influence_values[k] = {}
            for k2, sample_grad in self._sample_grad_hook.result_dict.items():
                if k2 not in self._influence_values[k]:
                    self._influence_values[k][k2] = 0
                self._influence_values[k][k2] += (
                    dot_product(
                        tensor_to(test_grad, device="cpu"),
                        tensor_to(sample_grad, device="cpu"),
                    )
                    * lr
                    / batch_size
                )
        get_logger().error("before reset")
        self._sample_grad_hook.reset_result()
        get_logger().error("after reset")

    def _after_execute(self, executor, **kwargs):
        if -1 in self._influence_values:
            assert len(self._influence_values) == 1
            self._influence_values = self._influence_values[-1]
        with open(
            os.path.join(
                executor.save_dir,
                "tracin.json",
            ),
            mode="wt",
            encoding="utf-8",
        ) as f:
            json.dump(self._influence_values, f)
        self._sample_grad_hook.release_queue()
