import torch.nn as nn
import torch
from loguru import logger
from ..model import BaseModel

class BaseEvaluator:
    def __init__(self,
                 model: BaseModel,
                 dataloader: torch.utils.data.DataLoader = None,
                 channel: int = 3,
                 input_size: int = 64,
                 decoder: nn.Module = None,
                 device='cpu'):
        """
        Args:
            model (BaseModel): flow-matching model to evaluate
        """
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.channel = channel
        self.input_size = input_size
        self.decoder = None

    def load_checkpoint(self, filepath):
        self.model.load_state_dict(torch.load(filepath))
        logger.info(f"Loaded checkpoint from {filepath}")

    def get_FID(self):
        pass

    def get_val_loss(self):
        pass

    def generate_images(self, target_dir, num_images=5000, batch_size=32, NFE=50):

        # T=1 (pure-noise) -> T=0 (clean image)
        sigmas = torch.linspace(1.0, 0.0, NFE + 1).to(self.device)
        
        noise = torch.randn(batch_size, self.channel, self.input_size, self.input_size).to(self.device)
        
        for i, sigma in enumerate(sigmas):
            dt = sigmas[i+1] - sigmas[i]
            
            with torch.no_grad():
                dx = self.model.forward()