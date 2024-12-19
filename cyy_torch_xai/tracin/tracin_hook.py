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
            for k2, sample_grad in self._sample_gradient_hook.result_dict.items():
                value = self._contribution.get_sample_contribution(
                    tracked_index=k2, test_index=k
                )
                self._contribution.set_sample_contribution(
                    tracked_index=k2,
                    test_index=k,
                    value=value + dot_product(test_grad, sample_grad) * lr / batch_size,
                )
        self._sample_gradient_hook.reset_result()

    def _after_execute(self, **kwargs: Any) -> None:
        executor = kwargs["executor"]
        os.makedirs(executor.save_dir, exist_ok=True)
        with open(
            os.path.join(
                executor.save_dir,
                "tracin.json",
            ),
            mode="w",
            encoding="utf-8",
        ) as f:
            self._contribution.dump(f)
        super()._after_execute(**kwargs)
