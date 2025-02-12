from abc import ABC, abstractmethod

class BaseDataset(ABC):
    @abstractmethod
    def get_dataloader(self, batch_size):
        pass

