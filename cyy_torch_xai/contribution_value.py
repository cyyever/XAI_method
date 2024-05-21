from cyy_torch_toolbox import IndicesType

from .typing import TrainingSubsetIndices


class ContributionValue:
    def __init__(self) -> None:
        self.__tracked_training_subsets: set[TrainingSubsetIndices] | None = None

    def set_tracked_training_subsets(
        self, tracked_training_subsets: set[TrainingSubsetIndices]
    ) -> None:
        assert tracked_training_subsets
        self.__tracked_training_subsets = tracked_training_subsets

    def set_tracked_training_indices(
        self, tracked_training_indices: IndicesType
    ) -> None:
        self.set_tracked_training_subsets(
            {frozenset({idx}) for idx in tracked_training_indices}
        )

    @property
    def tracked_training_subsets(self) -> set[TrainingSubsetIndices]:
        assert self.__tracked_training_subsets is not None
        return self.__tracked_training_subsets

    @property
    def tracked_training_indices(self) -> set[int]:
        indices = set()
        for subset in self.tracked_training_subsets:
            t = tuple(subset)
            assert len(t) == 1
            indices.add(t[0])
        return indices
