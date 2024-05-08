from typing import Iterable, TypeAlias

import torch
from cyy_torch_toolbox.typing import TensorDict

OptionalTensor: TypeAlias = torch.Tensor | None
OptionalTensorDict: TypeAlias = TensorDict | None
IndicesType: TypeAlias = Iterable[int]
OptionalIndicesType: TypeAlias = IndicesType | None
