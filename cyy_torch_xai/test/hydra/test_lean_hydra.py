from cyy_torch_algorithm.retraining import DeterministicTraining
from cyy_torch_toolbox.default_config import Config
from cyy_torch_toolbox.ml_type import ExecutorHookPoint, MachineLearningPhase
from cyy_torch_xai.lean_hydra.lean_hydra import LeanHyDRA
from cyy_torch_xai.lean_hydra.lean_hydra_config import LeanHyDRAConfig


def test_lean_hydra() -> None:
    config = LeanHyDRAConfig(dataset_name="MNIST", model_name="LeNet5")
    config.hyper_parameter_config.epoch = 2
    # config.hyper_parameter_config.learning_rate_scheduler_name = "CosineAnnealingLR"
    # config.hyper_parameter_config.optimizer_name = "Adam"
    config.hyper_parameter_config.optimizer_name = "SGD"
    config.hyper_parameter_config.learning_rate = 0.01
    # deterministic_training = DeterministicTraining(config)
    trainer = config.create_deterministic_trainer()
    trainer.train()
    res = config.recreate_trainer_and_hook()
    trainer = res["trainer"]
    hydra_obj = res["hook"]

    hydra_obj.set_computed_indices([0, 1])
    trainer.train()
    assert id(hydra_obj.optimizer) == id(trainer.get_optimizer())
    print(hydra_obj.get_contribution())


def test_api() -> None:
    config = Config(dataset_name="MNIST", model_name="LeNet5")
    config.hyper_parameter_config.epoch = 2
    config.hyper_parameter_config.learning_rate_scheduler_name = "CosineAnnealingLR"
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
        model_evaluator=trainer.model_evaluator,
        optimizer=trainer.get_optimizer(),
        test_gradient=test_gradient,
        training_set_size=len(trainer.dataset_util),
    )

    def set_optimizer(*args, **kwargs) -> None:
        hydra_obj.optimizer = trainer.get_optimizer()

    trainer.append_named_hook(
        ExecutorHookPoint.BEFORE_EXECUTE, "set_optimizer", set_optimizer
    )
    trainer.append_named_hook(
        ExecutorHookPoint.AFTER_FORWARD, "iterate", hydra_obj.iterate
    )
    trainer.append_named_hook(
        ExecutorHookPoint.CANCEL_FORWARD, "cancel_forward", hydra_obj.cancel_forward
    )
    hydra_obj.set_computed_indices([0, 1])
    trainer.train()
    assert id(hydra_obj.optimizer) == id(trainer.get_optimizer())
    print(hydra_obj.get_contribution())
