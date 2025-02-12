import torch
import torch.nn as nn
import torch.optim as optim
from abc import ABC, abstractmethod
from .utils import get_class
from .dataset import BaseDataset
from .model import BaseModel
from .trainer import BaseTrainer


class ExperimentFactory(ABC):
    @abstractmethod
    def create_model(self) -> BaseModel:
        pass

    @abstractmethod
    def create_dataset(self) -> BaseDataset:
        pass

    @abstractmethod
    def create_trainer(self, model: BaseModel, dataloader) -> BaseTrainer:
        pass


class GenericExperimentFactory(ExperimentFactory):
    def create_model(self, cfg) -> BaseModel:
        model_class = get_class(cfg.model._target_)
        model_params = cfg.model.get('params', {})
        return model_class(**model_params)
    
    def create_dataset(self, cfg) -> BaseDataset:
        dataset_class = get_class(cfg.dataset._target_)
        dataset_params = cfg.dataset.get('params', {})
        dataset_root = cfg.dataset.get('root', './data')
        dataset = dataset_class(root=dataset_root, **dataset_params)
        return dataset.get_dataloader(cfg.dataset.batch_size)
    
    def create_trainer(self, model: BaseModel, dataloader, cfg) -> BaseTrainer:
        trainer_class = get_class(cfg.trainer._target_)
        trainer_params = cfg.trainer.get('params', {})
        return trainer_class(
            model=model,
            dataloader=dataloader,
            device=cfg.trainer.device,
            num_epochs=cfg.trainer.num_epochs,
            **trainer_params
        )