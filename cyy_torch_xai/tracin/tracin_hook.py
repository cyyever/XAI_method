import json
import os
from typing import Any

import torch
from cyy_torch_toolbox.tensor import dot_product

from .base_hook import TracInBaseHook


class TracInHook(TracInBaseHook):

    def _after_batch(self, batch_size: int, **kwargs) -> None:
        optimizer = self._get_optimizer(**kwargs)
        assert len(optimizer.param_groups) == 1
        if not isinstance(optimizer, torch.optim.SGD):
            raise RuntimeError("optimizer is not SGD")
        lr = optimizer.param_groups[0]["lr"]

        assert self.test_gradients
        for k, test_grad in self.test_gradients.items():
            if k not in self._influence_values:
                self._influence_values[k] = {}
            for k2, sample_grad in self._sample_gradient_hook.result_dict.items():
                if k2 not in self._influence_values[k]:
                    self._influence_values[k][k2] = 0
                self._influence_values[k][k2] += (
                    dot_product(test_grad, sample_grad) * lr / batch_size
                )
        self._sample_gradient_hook.reset_result()

    def _after_execute(self, executor, **kwargs: Any) -> None:
        influence_values: dict = self._influence_values
        if -1 in self._influence_values:
            assert len(self._influence_values) == 1
            influence_values = self._influence_values[-1]
        os.makedirs(executor.save_dir, exist_ok=True)
        with open(
            os.path.join(
                executor.save_dir,
                "tracin.json",
            ),
            mode="wt",
            encoding="utf-8",
        ) as f:
            json.dump(influence_values, f)
        super()._after_execute(executor=executor, **kwargs)
