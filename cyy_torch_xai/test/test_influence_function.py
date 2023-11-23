from cyy_torch_toolbox.default_config import Config
from cyy_torch_xai.influence_function import compute_influence_function


def test_IF() -> None:
    config = Config(dataset_name="MNIST", model_name="LeNet5")
    config.hyper_parameter_config.epoch = 5
    config.hyper_parameter_config.learning_rate= 0.01
    trainer = config.create_trainer()
    trainer.train()
    compute_influence_function(
        trainer,
        computed_indices=[1, 2],
        inverse_hvp_arguments={"scale": 100000000, "epsilon": 0.03, "repeated_num": 1},
    )
