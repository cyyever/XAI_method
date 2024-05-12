import importlib

from cyy_torch_toolbox.concurrency import TorchProcessPool
from cyy_torch_xai.hydra.hydra_config import HyDRAConfig

has_cyy_torch_vision: bool = importlib.util.find_spec("cyy_torch_vision") is not None


def hydra_train() -> None:
    import cyy_torch_vision  # noqa: F401

    config = HyDRAConfig(dataset_name="MNIST", model_name="LeNet5")
    config.hyper_parameter_config.epoch = 1
    config.hyper_parameter_config.optimizer_name = "SGD"
    config.hyper_parameter_config.learning_rate = 0.01
    res = config.create_trainer()
    trainer = res["trainer"]
    hydra_obj = res["hook"]
    hydra_obj.set_computed_indices([0, 1])
    trainer.train()


def test_hydra() -> None:
    if not has_cyy_torch_vision:
        return

    pool = TorchProcessPool()
    pool.submit(hydra_train)
    pool.wait_results()
