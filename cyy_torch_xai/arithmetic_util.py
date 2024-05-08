import traceback
from typing import TypeAlias

import torch
from cyy_naive_lib.log import log_error

OptionalTensor: TypeAlias = torch.Tensor | None


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
        if res is None:
            res = a
        else:
            res = res + a
    return res


def optional_subtraction(a: OptionalTensor, b: OptionalTensor) -> OptionalTensor:
    if a is None:
        if b is None:
            return None
        return -b
    if b is None:
        return a
    return a - b


def optional_multiplication(*args: OptionalTensor) -> OptionalTensor:
    res = None
    for a in args:
        if a is None:
            return None
        if res is None:
            res = a
        else:
            res = res * a
    return res


def optional_division(
    a: OptionalTensor, b: torch.Tensor, epsilon: float
) -> OptionalTensor:
    if a is None:
        return None
    if epsilon is None:
        return a / b
    return a / (b + epsilon)
