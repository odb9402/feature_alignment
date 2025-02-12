import torch.nn as nn
from abc import ABC, abstractmethod

class BaseModel(ABC, nn.Module):
    @abstractmethod
    def forward(self, x):
        pass
