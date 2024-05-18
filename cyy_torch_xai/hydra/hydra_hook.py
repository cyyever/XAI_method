import json
import os
import pickle
import traceback
from typing import Any

import torch
from cyy_naive_lib.algorithm.sequence_op import split_list_to_chunks
from cyy_naive_lib.log import get_logger
from cyy_naive_lib.time_counter import TimeCounter
from cyy_torch_algorithm.data_structure.synced_tensor_dict import \
    SyncedTensorDict
from cyy_torch_toolbox import (ExecutorHookPoint, OptionalTensor, Trainer,
                               cat_tensor_dict, tensor_to)

from ..util import get_test_gradient
from .base_hook import BaseHook


class HyDRAHook(BaseHook):
    def __init__(self, cache_size, **kwargs) -> None:
        super().__init__(use_hessian=kwargs.get("use_hessian", False))
        self._cache_size: int = cache_size
        self._trainer: None | Trainer = None

        self._delayed_approximation_computations: dict = {}
        self.__hyper_parameter_size: None | int = None

        self._hessian_hyper_gradient_dict: SyncedTensorDict | None = None
        self._hessian_computation_arguments: dict = {}
        self.__hvp_arguments: dict = {}
        self.use_approximation = False

        use_approximation: None | bool = kwargs.get("use_approximation", None)
        if use_approximation is None:
            self.use_approximation = not self.use_hessian
        else:
            self.use_approximation = use_approximation

        self._approx_hyper_gradient_dict: SyncedTensorDict | None = None

    def _before_batch(self, executor, inputs, targets, **kwargs):
        trainer = executor
        if self._trainer is None:
            self._trainer = trainer

        if self.use_hessian:
            assert not self._hessian_computation_arguments
            self._hessian_computation_arguments = {}
            self.__hvp_arguments = {
                "executor": trainer,
                "inputs": inputs,
                "targets": targets,
            }

    @property
    def trainer(self) -> Trainer:
        assert self._trainer is not None
        return self._trainer

    @property
    def sample_gradient_dict(self) -> dict:
        return self._sample_gradient_hook.result_dict

    def __get_save_dir(self, trainer: Trainer) -> str:
        save_dir = trainer.save_dir
        assert save_dir is not None
        save_dir = os.path.join(save_dir, "HyDRA")
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    def _do_delayed_computation(
        self,
        use_approximation: bool,
        index: int,
        hessian_vector_product: OptionalTensor = None,
    ) -> None:
        raise NotImplementedError()

    def _before_execute(self, **kwargs) -> None:
        super()._before_execute(**kwargs)
        trainer = kwargs["executor"]
        with open(
            os.path.join(self.__get_save_dir(trainer), "tracking_indices.json"),
            encoding="utf8",
            mode="wt",
        ) as f:
            json.dump(list(self.computed_indices), f)
        if self.use_hessian:
            get_logger().info("use hessian to compute hyper-gradients")
            self._hessian_hyper_gradient_dict = HyDRAHook.create_hypergradient_dict(
                cache_size=self._cache_size,
                storage_dir=os.path.join(
                    self.__get_save_dir(trainer),
                    "hessian_hyper_gradient_dir",
                ),
            )
            get_logger().debug(
                "use hessian_hyper_gradient_mom_dir:%s",
                os.path.abspath(self._hessian_hyper_gradient_dict.get_storage_dir()),
            )
        if self.use_approximation:
            self._approx_hyper_gradient_dict = HyDRAHook.create_hypergradient_dict(
                cache_size=self._cache_size,
                storage_dir=os.path.join(
                    self.__get_save_dir(trainer),
                    "approximation_hyper_gradient_dir",
                ),
            )
            get_logger().info(
                "use approx dict:%s",
                os.path.abspath(self._approx_hyper_gradient_dict.get_storage_dir()),
            )
            self._delayed_approximation_computations = {}
            trainer.prepend_named_hook(
                hook_point=ExecutorHookPoint.BEFORE_BATCH,
                name="prepare_hook",
                fun=self.__prepare_hook,
                stripable=True,
            )

    def _after_execute(self, **kwargs):
        get_logger().info("end hyper-gradient tracking")
        trainer = kwargs["executor"]
        trainer.remove_named_hook(name="prepare_hook")
        if self.use_approximation:
            self.__save_hyper_gradients(
                trainer,
                use_approximation=True,
            )
        if self.use_hessian:
            self.__save_hyper_gradients(
                trainer,
                use_approximation=False,
            )
        super()._after_execute(**kwargs)

    @classmethod
    def create_hypergradient_dict(
        cls,
        cache_size,
        storage_dir,
    ):
        tensor_dict = SyncedTensorDict.create(
            key_type=int,
            cache_size=cache_size,
            storage_dir=storage_dir,
        )
        return tensor_dict

    def __prepare_hook(self, sample_indices: list[torch.Tensor], **kwargs: Any) -> None:
        if self.use_approximation:
            instance_indices: set[int] = {idx.data.item() for idx in sample_indices}
            batch_gradient_indices: set[int] = instance_indices & self.computed_indices
            if batch_gradient_indices:
                self._get_hyper_gradient_dict(self.use_approximation).prefetch(
                    batch_gradient_indices
                )

    def _set_hyper_gradient_tensors(self, index, use_approximation, *tensors) -> None:
        if self.__hyper_parameter_size is None:
            self.__hyper_parameter_size = tensors[0].shape[0]
        self._get_hyper_gradient_dict(use_approximation)[index] = torch.cat(tensors)

    def _decode_hyper_gradient_tensors(self, tensor) -> tuple:
        assert self.__hyper_parameter_size is not None
        return torch.split(tensor, self.__hyper_parameter_size)

    def _get_hyper_gradient_tensors(
        self, index: int, use_approximation: bool, none_num: int = 1
    ) -> tuple:
        data = self._get_hyper_gradient_dict(use_approximation)[index]
        if data is None:
            return (None,) * none_num
        return self._decode_hyper_gradient_tensors(data)

    def _get_hyper_gradient_dict(self, use_approximation: bool) -> SyncedTensorDict:
        res = (
            self._approx_hyper_gradient_dict
            if use_approximation
            else self._hessian_hyper_gradient_dict
        )
        assert res is not None
        return res

    def _do_all_delayed_computation(self) -> None:
        if self.use_approximation:
            delayed_keys = list(self._delayed_approximation_computations.keys())
            assert delayed_keys
            for chunk in split_list_to_chunks(delayed_keys, self._cache_size):
                self._get_hyper_gradient_dict(True).prefetch(chunk)
                for k in chunk:
                    get_logger().debug(
                        "do _delayed_approximation_computations for %s", k
                    )
                    self._do_delayed_computation(True, k)
            return

    def _do_computation_with_hessian(self) -> None:
        for chunk in split_list_to_chunks(
            list(self.computed_indices), self._cache_size
        ):
            hessian_vector_product_dict = self._get_hvp(chunk)
            for index in chunk:
                hessian_vector_product = hessian_vector_product_dict.get(index, None)
                if hessian_vector_product is not None:
                    hessian_vector_product = hessian_vector_product.to(
                        self.trainer.device
                    )
                    self._check_overflow_and_underflow(hessian_vector_product)
                self._do_delayed_computation(False, index, hessian_vector_product)

    def _check_overflow_and_underflow(self, tensor: OptionalTensor) -> None:
        if tensor is None:
            return
        if torch.any(torch.isnan(tensor)):
            get_logger().error("find nan tensor %s", tensor.cpu())
            get_logger().error("traceback:%s", str(traceback.extract_stack(limit=10)))
            raise AssertionError()
        if torch.any(torch.isinf(tensor)):
            get_logger().error("find inf tensor %s", tensor.cpu())
            get_logger().error("traceback:%s", str(traceback.extract_stack(limit=10)))
            raise AssertionError()

    def __save_hyper_gradients(self, trainer, use_approximation):
        test_gradient = get_test_gradient(trainer)
        contribution = {}
        get_logger().info("begin do _do_all_delayed_computation")
        self._do_all_delayed_computation()
        get_logger().info("end do _do_all_delayed_computation")
        tensor_dict = self._get_hyper_gradient_dict(use_approximation)
        test_gradient = tensor_to(cat_tensor_dict(test_gradient), device="cpu")
        for index, value in tensor_dict.items():
            hyper_gradient = self._decode_hyper_gradient_tensors(value)[0]
            contribution[index] = (
                -(test_gradient @ hyper_gradient) / self.training_set_size
            ).item()
            tensor_dict[index] = hyper_gradient
        assert contribution
        json_name = "hessian_hydra_contribution.json"
        if use_approximation:
            json_name = "approx_hydra_contribution.json"
        with open(
            os.path.join(self.__get_save_dir(trainer), json_name),
            mode="wt",
            encoding="utf-8",
        ) as f:
            json.dump(contribution, f)
        with open(
            os.path.join(self.__get_save_dir(trainer), "training_set_size"), "wb"
        ) as f:
            pickle.dump(self.training_set_size, f)

    def _get_hvp(self, chunk) -> dict:
        assert self._hessian_hyper_gradient_dict is not None
        self._hessian_hyper_gradient_dict.prefetch(chunk)
        hyper_gradients = []
        hyper_gradient_indices = []
        hessian_vector_product_dict: dict = {}
        for index in chunk:
            hyper_gradient = self.get_hyper_gradient(index, use_approximation=False)
            if hyper_gradient is not None:
                hyper_gradients.append(hyper_gradient)
                hyper_gradient_indices.append(index)
        if not hyper_gradients:
            return hessian_vector_product_dict
        with TimeCounter(log_prefix=f"hvp chunk size {len(hyper_gradients)}"):
            self.batch_hvp_hook.add_task(
                data=hyper_gradients,
                batch_index=0,
                **self.__hvp_arguments,
            )
            hessian_vector_products = self.batch_hvp_hook.result_dict
            hessian_vector_product_dict = {
                gradient_idx: hessian_vector_products[product_idx]
                for product_idx, gradient_idx in enumerate(hyper_gradient_indices)
            }
            self.batch_hvp_hook.reset_result()
            assert not self.batch_hvp_hook.result_dict
            return hessian_vector_product_dict

    def get_hyper_gradient(self, index, use_approximation):
        return self._get_hyper_gradient_tensors(index, use_approximation)[0]

    def foreach_hyper_gradient(self, use_approximation: bool, callback):
        self._do_all_delayed_computation()
        hyper_gradient_dir = self._get_hyper_gradient_dict(use_approximation)
        for index, _ in hyper_gradient_dir.items():
            hyper_gradient = self.get_hyper_gradient(index, use_approximation)
            callback(index, hyper_gradient)

    def foreach_approx_and_hessian_hyper_gradient(self, callback):
        assert self.use_approximation and self.use_hessian
        self._do_all_delayed_computation()
        approximation_hyper_gradient_dir = self._get_hyper_gradient_dict(True)
        for index, _ in approximation_hyper_gradient_dir.items():
            approx_hyper_gradient = self.get_hyper_gradient(index, True)
            hessian_hyper_gradient = self.get_hyper_gradient(index, False)
            callback(index, approx_hyper_gradient, hessian_hyper_gradient)
