import copy

import torch
from cyy_naive_lib.log import get_logger
from cyy_torch_algorithm.computation.batch_hvp.batch_hvp_hook import \
    BatchHVPHook
from cyy_torch_toolbox.inferencer import Inferencer
from cyy_torch_toolbox.ml_type import ExecutorHookPoint, StopExecutingException
from cyy_torch_toolbox.tensor import cat_tensor_dict, tensor_to


def stochastic_inverse_hessian_vector_product(
    inferencer: Inferencer,
    vectors: list,
    repeated_num: int = 1,
    max_iteration: int = 1000,
    dampling_term: float = 0,
    scale: float = 1,
    epsilon: float = 0.0001,
) -> torch.Tensor:
    get_logger().info(
        "repeated_num is %s,max_iteration is %s,dampling term is %s,scale is %s,epsilon is %s",
        repeated_num,
        max_iteration,
        dampling_term,
        scale,
        epsilon,
    )

    def iteration(inferencer, vectors) -> torch.Tensor:
        iteration_num = 0
        hook = BatchHVPHook()

        tmp_inferencer = copy.deepcopy(inferencer)
        tmp_inferencer.disable_hook("logger")
        tmp_inferencer.disable_hook("performance_metric_logger")
        cur_products = copy.deepcopy(vectors)
        hook.set_data_fun(lambda: cur_products)

        results: None | torch.Tensor = None

        def compute_product(epoch, **kwargs) -> None:
            nonlocal cur_products
            nonlocal results
            nonlocal iteration_num
            nonlocal vectors
            nonlocal hook
            assert len(hook.result_dict) == len(vectors)
            # + (1 - dampling_term) * cur_products

            next_products: list = []
            for idx, vector in enumerate(vectors):
                next_products.append({})
                for k in vector:
                    next_products[idx][k] = (
                        vector[k]
                        + cur_products[idx][k]
                        - tensor_to(
                            hook.result_dict[idx][k], device=tmp_inferencer.device
                        )
                        / scale
                    )
            hook.reset_result()
            diffs = torch.tensor(
                [
                    torch.dist(cat_tensor_dict(a), cat_tensor_dict(b))
                    for a, b in zip(cur_products, next_products)
                ]
            )
            get_logger().info(
                "diffs is %s, epsilon is %s, epoch is %s, iteration is %s, max_iteration is %s, scale %s",
                diffs,
                epsilon,
                epoch,
                iteration_num,
                max_iteration,
                scale,
            )
            cur_products = next_products
            iteration_num += 1
            if (
                (diffs <= epsilon).all().bool() and epoch > 1
            ) or iteration_num >= max_iteration:
                results = torch.stack(
                    [cat_tensor_dict(p) / scale for p in cur_products]
                )
                raise StopExecutingException()
            hook.set_data_fun(lambda: cur_products)

        tmp_inferencer.append_hook(hook)
        tmp_inferencer.append_named_hook(
            hook_point=ExecutorHookPoint.AFTER_FORWARD,
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
        return results.cpu()

    product_list: torch.Tensor = torch.stack(
        [iteration(inferencer, vectors) for _ in range(repeated_num)]
    )
    std, mean = torch.std_mean(product_list, dim=0)
    get_logger().info("std is %s", torch.norm(std, p=2))
    return mean
