import traceback

import torch
from cyy_naive_lib.log import log_error
from cyy_torch_toolbox import OptionalTensor


def check_overflow_and_underflow(tensor: torch.Tensor) -> None:
    if tensor is None:
        return
    if torch.any(torch.isnan(tensor)):
        log_error("traceback:%s", str(traceback.extract_stack(limit=10)))
        raise AssertionError(f"find nan tensor {tensor.cpu()}")
    if torch.any(torch.isinf(tensor)):
        log_error("traceback:%s", str(traceback.extract_stack(limit=10)))
        raise AssertionError(f"find inf tensor {tensor.cpu()}")


def optional_addition(*args: OptionalTensor) -> OptionalTensor:
    res = None
    for a in args:
        if a is None:
            continue
        res = a if res is None else res + a
    return res


def optional_subtraction(a: OptionalTensor, b: OptionalTensor) -> OptionalTensor:
    if a is None:
        if b is None:
            return None
        return -b
    if b is None:
        return a
    return a - b


def optional_multiplication(a: OptionalTensor, *args: float) -> OptionalTensor:
    if a is None:
        return None
    res = a
    for arg in args:
        res = res * arg
    return res


def optional_division(
    a: OptionalTensor, b: torch.Tensor, epsilon: float | None
) -> OptionalTensor:
    if a is None:
        return None
    if epsilon is None:
        return a / b
    return a / (b + epsilon)
