#!/usr/bin/env python3

from cyy_ml_if.lean_hydra.lean_hydra import LeanHyDRA
from cyy_torch_algorithm.retraining import DeterministicTraining
from cyy_torch_toolbox.default_config import DefaultConfig
from cyy_torch_toolbox.ml_type import (MachineLearningPhase,
                                       ModelExecutorHookPoint)


def test_api():
    config = DefaultConfig(dataset_name="MNIST", model_name="LeNet5")
    config.hyper_parameter_config.epoch = 10
    config.cache_transforms = "cpu"
    config.hyper_parameter_config.learning_rate_scheduler = "CosineAnnealingLR"
    # config.hyper_parameter_config.optimizer_name = "Adam"
    config.hyper_parameter_config.optimizer_name = "SGD"
    config.hyper_parameter_config.learning_rate = 0.01
    deterministic_training = DeterministicTraining(config)
    trainer = deterministic_training.create_deterministic_trainer()
    trainer.train()
    test_gradient = trainer.get_inferencer(
        phase=MachineLearningPhase.Test
    ).get_gradient()
    trainer = deterministic_training.recreate_trainer()
    hydra_obj = LeanHyDRA(
        model=trainer.model,
        loss_function=trainer.loss_fun,
        optimizer=trainer.get_optimizer(),
        test_gradient=test_gradient,
        training_set_size=len(trainer.dataset_util),
    )

    def set_optimizer(*args, **kwargs):
        hydra_obj.optimizer = trainer.get_optimizer()

    trainer.append_named_hook(
        ModelExecutorHookPoint.BEFORE_EXECUTE, "set_optimizer", set_optimizer
    )
    trainer.append_named_hook(
        ModelExecutorHookPoint.AFTER_FORWARD, "iterate", hydra_obj.iterate
    )
    hydra_obj.set_computed_indices([0, 1])
    trainer.train()
    assert id(hydra_obj.optimizer) == id(trainer.get_optimizer())
    print(hydra_obj.get_contribution())
