from typing import Any

from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox import Hook, IndicesType, Trainer


class SampleBaseHook(Hook):
    def __init__(self) -> None:
        super().__init__(stripable=True)
        self.__computed_indices: set[int] | None = None
        self.__training_set_size: int | None = None

    @property
    def computed_indices(self) -> set[int]:
        assert self.__computed_indices is not None
        return self.__computed_indices

    def set_computed_indices(self, computed_indices: IndicesType) -> None:
        assert computed_indices
        self.__computed_indices = set(computed_indices)

    @property
    def training_set_size(self) -> int:
        assert self.__training_set_size is not None
        return self.__training_set_size

    def _before_execute(self, **kwargs: Any) -> None:
        if "executor" in kwargs:
            trainer = kwargs["executor"]
            assert isinstance(trainer, Trainer)
            self.__training_set_size = trainer.dataset_size
        else:
            self.__training_set_size = kwargs["training_set_size"]

        if self.__computed_indices is None:
            self.set_computed_indices(range(self.training_set_size))
        else:
            get_logger().info("only compute %s indices", len(self.computed_indices))
