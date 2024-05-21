import importlib

from cyy_torch_toolbox.config import Config
from cyy_torch_xai.tracin import TracInHook

has_cyy_torch_vision: bool = importlib.util.find_spec("cyy_torch_vision") is not None


def test_tracin() -> None:
    if not has_cyy_torch_vision:
        return

    import cyy_torch_vision  # noqa: F401

    config = Config(dataset_name="MNIST", model_name="LeNet5")
    config.hyper_parameter_config.epoch = 2
    config.hyper_parameter_config.optimizer_name = "SGD"
    config.hyper_parameter_config.learning_rate = 0.01
    trainer = config.create_trainer()
    trainer.append_hook(TracInHook())
    trainer.train()
