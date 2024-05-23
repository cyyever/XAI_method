import importlib

from cyy_torch_toolbox import Config
from cyy_torch_xai.representer_point import compute_representer_point_values

has_cyy_torch_vision: bool = importlib.util.find_spec("cyy_torch_vision") is not None


if has_cyy_torch_vision:

    def test_representer_point() -> None:
        import cyy_torch_vision  # noqa: F401

        config = Config(dataset_name="MNIST", model_name="LeNet5")
        config.hyper_parameter_config.epoch = 1
        config.hyper_parameter_config.learning_rate = 0.01
        trainer = config.create_trainer()
        trainer.train()
        contribution = compute_representer_point_values(
            trainer=trainer, test_indices=[1, 2], training_indices=[1, 2, 3, 4, 5]
        )
        print(contribution.normalized_values)
