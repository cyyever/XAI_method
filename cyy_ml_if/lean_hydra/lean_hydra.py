import torch
from cyy_torch_toolbox.model_with_loss import ModelWithLoss

from .lean_hydra_sgd_hook import LeanHyDRASGDHook


class LeanHyDRA:
    def __init__(
        self,
        model,
        loss_function,
        optimizer,
        test_gradient,
        training_set_size,
    ):
        if isinstance(optimizer, torch.optim.SGD):
            self._hook = LeanHyDRASGDHook(test_gradient=test_gradient)
        else:
            raise RuntimeError(f"unsupported optimizer type {type(optimizer)}")

        self.optimizer = optimizer
        self.model_with_loss = ModelWithLoss(model, loss_function)
        self._hook._before_execute(training_set_size=training_set_size)

    def set_computed_indices(self, computed_indices):
        self._hook.set_computed_indices(computed_indices=computed_indices)

    def iterate(self, batch_indexes, batch_input, batch_targets):
        batch_indexes = batch_indexes.tolist()
        self._hook.sample_gradient_hook.add_task(
            self.model_with_loss, batch_indexes, batch_input, batch_targets
        )
        self._hook._before_batch(optimizer=self.optimizer)
        self._hook._after_optimizer_step(
            batch_size=len(batch_indexes), step_skipped=False
        )

    def get_contribution(self, **kwargs):
        self._hook.sample_gradient_hook.release_queue()
        return self._hook.contributions

