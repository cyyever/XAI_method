import traceback
from typing import Callable

import torch
from cyy_naive_lib.log import get_logger
from cyy_torch_algorithm.computation.sample_gradient.sample_gradient_hook import \
    get_sample_gradient_dict
from cyy_torch_algorithm.data_structure.synced_tensor_dict import \
    SyncedTensorDict
from cyy_torch_toolbox.ml_type import MachineLearningPhase
from cyy_torch_toolbox.tensor import tensor_to
from cyy_torch_toolbox.trainer import Trainer


def compute_perturbation_gradient_difference(
    trainer: Trainer,
    perturbation_idx_fun: Callable,
    perturbation_fun: Callable,
    result_transform: Callable | None = None,
) -> dict | SyncedTensorDict:
    inferencer = trainer.get_inferencer(
        phase=MachineLearningPhase.Training, copy_model=True
    )
    trainer.offload_from_gpu()

    sample_to_perturbations: dict = {}

    def sample_selector(sample_index, sample_input):
        nonlocal sample_to_perturbations

        res = perturbation_idx_fun(sample_index=sample_index, sample_input=sample_input)
        if res:
            assert sample_index not in sample_to_perturbations
            sample_to_perturbations[sample_index] = res
            return True
        return False

    if result_transform is None:
        sample_dict = SyncedTensorDict.create(cache_size=128, key_type=None)
    else:
        sample_dict: dict = {}

    def collect_result(result_dict):
        nonlocal sample_dict
        nonlocal sample_to_perturbations
        for sample_idx, v in result_dict.items():
            v = tensor_to(v, device="cpu", non_blocking=True)
            for perturbation_idx in sample_to_perturbations[sample_idx]:
                if perturbation_idx not in sample_dict:
                    sample_dict[perturbation_idx] = v
                else:
                    sample_dict[perturbation_idx] = sample_dict[perturbation_idx] + v

    get_sample_gradient_dict(
        inferencer=inferencer,
        sample_selector=sample_selector,
        result_transform=result_transform,
        result_collection_fun=collect_result,
    )
    if result_transform is None:
        perturbation_dict = SyncedTensorDict.create(cache_size=128, key_type=None)
    else:
        perturbation_dict: dict = {}

    def collect_result2(result_dict):
        nonlocal perturbation_dict
        for k, v in result_dict.items():
            sample_index, perturbation_index = k
            v = tensor_to(v, device="cpu", non_blocking=True)
            if perturbation_index not in perturbation_dict:
                perturbation_dict[perturbation_index] = v
            else:
                perturbation_dict[perturbation_index] = (
                    perturbation_dict[perturbation_index] + v
                )

    def sample_selector2(sample_index, sample_input):
        res = perturbation_idx_fun(sample_index=sample_index, sample_input=sample_input)
        if res:
            return True
        return False

    get_sample_gradient_dict(
        inferencer=inferencer,
        sample_selector=sample_selector2,
        input_transform=perturbation_fun,
        result_transform=result_transform,
        result_collection_fun=collect_result2,
    )
    if result_transform is None:
        result = SyncedTensorDict.create(cache_size=128, key_type=None)
    else:
        result: dict = {}
    assert len(sample_dict) == len(perturbation_dict)
    for perturbation_idx in sample_dict.keys():
        tmp = sample_dict[perturbation_idx] - perturbation_dict[perturbation_idx]
        if isinstance(tmp, torch.Tensor):
            tmp = tmp.cpu()
        result[perturbation_idx] = tmp
    if result_transform is None:
        sample_dict.release()
        perturbation_dict.release()
    return result


def check_overflow_and_underflow(tensor):
    if tensor is None:
        return
    if torch.any(torch.isnan(tensor)):
        get_logger().error("find nan tensor %s", tensor.cpu())
        get_logger().error("traceback:%s", str(traceback.extract_stack(limit=10)))
        assert False
    if torch.any(torch.isinf(tensor)):
        get_logger().error("find inf tensor %s", tensor.cpu())
        get_logger().error("traceback:%s", str(traceback.extract_stack(limit=10)))
        assert False


def optional_addition(*args):
    res = None
    for a in args:
        if a is None:
            continue
        if res is None:
            res = a
        else:
            res = res + a
    return res


def optional_multiplication(*args):
    res = None
    for a in args:
        if a is None:
            return None
        if res is None:
            res = a
        else:
            res = res * a
    return res


def optional_division(a, b, epsilon):
    if a is None:
        return None
    if epsilon is None:
        return a / b
    return a / (b + epsilon)
