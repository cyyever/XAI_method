import torch.optim
from cyy_torch_toolbox import Config, MachineLearningPhase, Trainer, Hook
from cyy_torch_toolbox.dataset.sampler import DatasetSampler

from .hydra_adam_hook import HyDRAAdamHook
from .hydra_sgd_hook import HyDRASGDHook


class HyDRAConfig(Config):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.cache_size: int = 128
        self.tracking_percentage: float | None = None
        self.use_hessian: bool = False
        self.use_approximation: bool = True

    def create_trainer(self, **kwargs) -> tuple[Trainer, Hook]:
        trainer = super().create_trainer(**kwargs)

        optimizer = trainer.get_optimizer()
        if isinstance(optimizer, torch.optim.SGD):
            hydra_hook = HyDRASGDHook(
                cache_size=self.cache_size,
                use_hessian=self.use_hessian,
                use_approximation=self.use_approximation,
            )
        elif isinstance(optimizer, torch.optim.Adam):
            hydra_hook = HyDRAAdamHook(
                cache_size=self.cache_size,
                use_hessian=self.use_hessian,
                use_approximation=self.use_approximation,
            )
        else:
            raise NotImplementedError(f"Unsupported optimizer {type(optimizer)}")
        trainer.remove_optimizer()
        trainer.append_hook(hydra_hook)

        if self.tracking_percentage is not None:
            subset = DatasetSampler(
                trainer.dataset_collection.get_dataset_util(
                    phase=MachineLearningPhase.Training
                )
            ).iid_sample_indices(self.tracking_percentage)
            hydra_hook.set_computed_indices(subset)
        return trainer, hydra_hook
