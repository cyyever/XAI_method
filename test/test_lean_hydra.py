import cyy_torch_vision  # noqa: F401
from cyy_torch_toolbox.concurrency import TorchProcessPool
from cyy_torch_xai.lean_hydra import LeanHyDRAConfig, LeanHyDRAHook


def lean_hydra_train() -> None:
    config = LeanHyDRAConfig(dataset_name="MNIST", model_name="LeNet5")
    config.hyper_parameter_config.epoch = 2
    config.hyper_parameter_config.optimizer_name = "SGD"
    config.hyper_parameter_config.learning_rate = 0.01
    trainer = config.create_deterministic_trainer()
    trainer.train()
    trainer, hydra_hook = config.recreate_trainer_and_hook()
    assert isinstance(hydra_hook, LeanHyDRAHook)
    hydra_hook.set_computed_indices([0, 1])
    trainer.train()


def test_lean_hydra() -> None:
    pool = TorchProcessPool()
    pool.submit(lean_hydra_train)
    pool.wait_results()
