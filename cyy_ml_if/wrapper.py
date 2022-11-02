from typing import Callable

import torch
from cyy_torch_toolbox.dataset_collection import DatasetCollection
from cyy_torch_toolbox.hyper_parameter import HyperParameter
from cyy_torch_toolbox.inferencer import Inferencer
from cyy_torch_toolbox.ml_type import MachineLearningPhase, ModelType
from cyy_torch_toolbox.model_with_loss import ModelWithLoss


def wrap_trainer(
    training_dataset: torch.utils.data.Dataset,
    validation_dataset: torch.utils.data.Dataset,
    test_dataset: torch.utils.data.Dataset,
    model: torch.nn.Module,
    loss_fun: str | Callable,
) -> Inferencer:
    dc = DatasetCollection(
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
    )
    model_with_loss = ModelWithLoss(
        model=model, loss_fun=loss_fun, model_type=ModelType.Classification
    )
    return Inferencer(
        dataset_collection=dc,
        model_with_loss=model_with_loss,
        phase=MachineLearningPhase.Test,
        hyper_parameter=HyperParameter(
            epoch=None, batch_size=32, learning_rate=None, weight_decay=None
        ),
    )
