import json

from cyy_torch_toolbox import IndicesType

from .typing import SubsetIndices


class SubsetContribution:
    def __init__(self) -> None:
        self.__tracked_subsets: set[SubsetIndices] | None = None
        self.__values: dict[frozenset[int] | None, dict[frozenset[int], float]] = {}

    @property
    def values(self) -> dict:
        return self.__values

    def set_sample_contribution(
        self, tracked_index: int, value: float, test_index: int | None = None
    ) -> None:
        subset = self.__get_set(tracked_index)
        assert subset in self.tracked_subsets
        test_subset = self.__get_set(test_index)
        if test_subset not in self.__values:
            self.__values[test_subset] = {}
        self.__values[test_subset][subset] = value

    def get_sample_contribution(
        self,
        tracked_index: int,
        test_index: int | None = None,
        default_value: float = 0,
    ) -> float:
        subset = self.__get_set(tracked_index)
        assert subset in self.tracked_subsets
        test_subset = self.__get_set(test_index)
        if test_subset not in self.__values:
            return default_value
        return self.__values[test_subset].get(subset, default_value)

    def clear_contributions(self) -> None:
        self.__values.clear()

    def __get_set(self, index: int | None = None) -> frozenset | None:
        if index is None:
            return None
        return frozenset([index])

    def set_tracked_subsets(self, tracked_subsets: set[SubsetIndices]) -> None:
        assert tracked_subsets
        self.__tracked_subsets = tracked_subsets

    def set_tracked_indices(self, tracked_indices: IndicesType) -> None:
        self.set_tracked_subsets({frozenset({idx}) for idx in tracked_indices})

    @property
    def tracked_subsets(self) -> set[SubsetIndices]:
        assert self.__tracked_subsets is not None
        return self.__tracked_subsets

    @property
    def tracked_indices(self) -> set[int]:
        indices = set()
        for subset in self.tracked_subsets:
            t = tuple(subset)
            assert len(t) == 1
            indices.add(t[0])
        return indices

    def dump(self, file) -> None:
        def transform_key(indices: SubsetIndices | None) -> int | str:
            if indices is None:
                return "None"
            if len(indices) != 1:
                return str(indices)
            return list(indices)[0]

        data = {
            transform_key(k): {transform_key(k2): v2 for k2, v2 in v.items()}
            for k, v in self.__values.items()
        }
        if len(data) == 1:
            data = next(iter(data.values()))
        assert data

        json.dump(data, file)
