import json
import os

from cyy_ml_if.hydra.base_hook import BaseHook


class LeanHyDRAHook(BaseHook):
    def __init__(self, test_gradient, **kwargs):
        super().__init__(**kwargs)
        self.__test_gradient = test_gradient.cpu()
        self._sample_gradient_hook.set_result_transform(self._gradient_dot_product)
        if self._batch_hvp_hook is not None:
            self._batch_hvp_hook.set_vectors([self.__test_gradient])

    def _gradient_dot_product(self, result, **kwargs):
        return self.__test_gradient.dot(result.cpu()).item()

    def _after_execute(self, **kwargs) -> None:
        assert self._contributions is not None
        assert self._contributions.shape[0] == self._training_set_size
        save_dir = "."
        if "model_executor" in kwargs:
            trainer = kwargs["model_executor"]
            save_dir = os.path.join(trainer.save_dir, "lean_HyDRA")
            os.makedirs(save_dir, exist_ok=True)

        with open(
            os.path.join(save_dir, "lean_hydra_contribution.json"),
            mode="wt",
            encoding="utf-8",
        ) as f:
            contributions = self._contributions.cpu().tolist()
            json.dump({idx: contributions[idx] for idx in self._computed_indices}, f)
        super()._after_execute(**kwargs)
