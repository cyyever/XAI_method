import copy
import functools
import math

import torch
from cyy_naive_lib.log import get_logger
from cyy_torch_algorithm.computation.batch_hvp.batch_hvp_hook import \
    BatchHVPHook
from cyy_torch_toolbox import (ExecutorHookPoint, Inferencer,
                               StopExecutingException)
from cyy_torch_toolbox.tensor import tensor_to
from cyy_torch_toolbox.typing import TensorDict


def __vector_diff(a, b) -> float:
    product = 0
    for k, v in a.items():
        c = (v - b[k]).view(-1)

        product += c.dot(c).item()
    return math.sqrt(product)


def __result_transform(scale, data_index: int, result, data: dict) -> dict:
    return {k: v - result[k] / scale for k, v in data.items()}


default_inverse_hvp_arguments: dict[str, float | int] = {
    "dampling_term": 0.01,
    "scale": 100000,
    "epsilon": 0.03,
    "repeated_num": 3,
    "max_iteration": 1000,
}


def stochastic_inverse_hessian_vector_product(
    inferencer: Inferencer,
    vectors: list,
    dampling_term: float | None = None,
    scale: float | None = None,
    epsilon: float | None = None,
    repeated_num: int | None = None,
    max_iteration: int | None = None,
) -> list[TensorDict]:
    if dampling_term is None:
        dampling_term = default_inverse_hvp_arguments["dampling_term"]
    if scale is None:
        scale = default_inverse_hvp_arguments["scale"]
    if epsilon is None:
        epsilon = default_inverse_hvp_arguments["epsilon"]
    if max_iteration is None:
        max_iteration = int(default_inverse_hvp_arguments["max_iteration"])
    if repeated_num is None:
        repeated_num = int(default_inverse_hvp_arguments["repeated_num"])
    get_logger().info(
        "repeated_num is %s,max_iteration is %s,dampling term is %s,scale is %s,epsilon is %s",
        repeated_num,
        max_iteration,
        dampling_term,
        scale,
        epsilon,
    )

    def iteration(inferencer, vectors) -> list[dict]:
        iteration_num = 0
        hook = BatchHVPHook()

        tmp_inferencer = copy.deepcopy(inferencer)
        tmp_inferencer.disable_hook("logger")
        tmp_inferencer.disable_hook("performance_metric_logger")
        cur_products = copy.deepcopy(vectors)
        hook.set_data(cur_products)

        hook.set_result_transform(functools.partial(__result_transform, scale))

        results: None | list[dict] = None

        def compute_product(epoch, **kwargs) -> None:
            nonlocal cur_products
            nonlocal results
            nonlocal iteration_num
            nonlocal vectors
            nonlocal hook
            nonlocal max_iteration
            nonlocal repeated_num
            assert len(hook.result_dict) == len(vectors)

            next_products: list = []
            for idx, vector in enumerate(vectors):
                next_products.append({})
                for k in vector:
                    next_products[idx][k] = vector[k] + tensor_to(
                        hook.result_dict[idx][k],
                        device=vector[k].device,
                        non_blocking=True,
                    )
            hook.reset_result()
            max_diff = max(
                __vector_diff(a, b) for a, b in zip(cur_products, next_products)
            )
            get_logger().info(
                "max diff is %s, epsilon is %s, epoch is %s, iteration is %s, max_iteration is %s, scale %s",
                max_diff,
                epsilon,
                epoch,
                iteration_num,
                max_iteration,
                scale,
            )
            cur_products = next_products
            iteration_num += 1
            if (max_diff <= epsilon and epoch > 1) or iteration_num >= max_iteration:
                results = [
                    {k: v / scale}
                    for product in cur_products
                    for k, v in product.items()
                ]
                raise StopExecutingException()
            hook.set_data(cur_products)

        tmp_inferencer.append_hook(hook)
        tmp_inferencer.append_named_hook(
            hook_point=ExecutorHookPoint.AFTER_BATCH,
            name="compute_product",
            fun=compute_product,
        )
        epoch = 1
        while results is None:
            get_logger().debug(
                "stochastic_inverse_hessian_vector_product epoch is %s", epoch
            )
            normal_stop = tmp_inferencer.inference(use_grad=False, epoch=epoch)
            if not normal_stop:
                break
            epoch += 1
        del cur_products
        hook.release_queue()
        assert results is not None
        return results

    product_list: list[list[dict]] = [
        iteration(inferencer, vectors) for _ in range(repeated_num)
    ]
    if repeated_num == 1:
        return product_list[0]
    tmp: list[dict] = [{}] * len(vectors)
    for products in product_list:
        for idx, product in enumerate(products):
            if not tmp[idx]:
                tmp[idx] = {k: [v] for k, v in product.items()}
            else:
                for k, v in product.items():
                    tmp[idx][k].append(v)
    for product in tmp:
        for k, v in product:
            std, mean = torch.std_mean(v, dim=0)
            product[k] = mean
            get_logger().info("std is %s", torch.linalg.vector_norm(std))
    return tmp
