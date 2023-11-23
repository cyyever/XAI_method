from cyy_torch_toolbox.default_config import Config
from cyy_torch_xai.influence_function import compute_influence_function


def test_IF() -> None:
    config = Config(dataset_name="MNIST", model_name="LeNet5")
    config.hyper_parameter_config.epoch = 10
    trainer = config.create_trainer()
    trainer.train()
    compute_influence_function(trainer, computed_indices=[1, 2])
