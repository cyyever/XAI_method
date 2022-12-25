#!/usr/bin/env python3

from cyy_ml_if.lean_hydra.lean_hydra import LeanHyDRA
from cyy_torch_toolbox.default_config import DefaultConfig
from cyy_torch_toolbox.hooks.add_index_to_dataset import AddIndexToDataset
from cyy_torch_toolbox.ml_type import (MachineLearningPhase,
                                       ModelExecutorHookPoint)


def test_api():
    config = DefaultConfig(dataset_name="MNIST", model_name="LeNet5")
    config.hyper_parameter_config.epoch = 1
    config.hyper_parameter_config.learning_rate = 0.01
    trainer = config.create_trainer()
    trainer.append_hook(AddIndexToDataset())
    trainer.train()
    test_gradient = trainer.get_inferencer(
        phase=MachineLearningPhase.Test
    ).get_gradient()
    hydra_obj = LeanHyDRA(
        model=trainer.model,
        loss_function=trainer.loss_fun,
        optimizer=trainer.get_optimizer(),
        test_gradient=test_gradient,
        training_set_size=len(trainer.dataset_util),
    )
    hydra_obj.set_computed_indices([0, 1])
    trainer.append_named_hook(
        ModelExecutorHookPoint.AFTER_FORWARD, "iterate", hydra_obj.iterate
    )
    trainer.train()
    print(hydra_obj.get_contribution())
