import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Union
from loguru import logger  # Added loguru import

from diffusers.utils.torch_utils import randn_tensor
from diffusers.models import AutoencoderKL

from . import BaseTrainer
from ..model import BaseModel

class FlowMatchingTrainer(BaseTrainer):
    
    MIN_STD: float = 0.0  # minimum size of std for the flow matching
    CLAMP_CONTINUOUS_TIME: float = 0.0
    
    def __init__(self, model: BaseModel, dataloader, num_epochs=10, device='cpu', lr=1e-4, img_encoder="sd3"):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.num_epochs = num_epochs
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        if img_encoder == "sd3":
            self.vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", subfolder="vae")
        else:
            self.vae = None

        logger.info("Initialized FlowMatchingTrainer with num_epochs: {}, device: {}, learning_rate: {}, image_encoder: {}",
                    num_epochs, device, lr, img_encoder)

    def train(self):
        self.model.train()
        loss_history = []

        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            logger.info("Starting epoch {}/{}", epoch + 1, self.num_epochs)

            for step, (image, label) in enumerate(self.dataloader, 1):
                image = image.to(self.device)
                self.optimizer.zero_grad()

                latents = self.maybe_encode_image(image)
                cond_input = self.maybe_get_conditional_input()
                noise, noise_latents, t = self.get_noisy_input(latents)

                timesteps = self.discritize_timestep(t, self.n_timesteps)

                # Forward pass through the model
                preds = self.model(noise_latents, timesteps, cond_input)
                
                # Rectified flow matching loss
                loss = self.rectified_flow_loss(latents, noise, t, preds, use_weighting=self.use_weighting)
                
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                
                # Log each step's loss and relevant details
                logger.info("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}",
                            epoch + 1, self.num_epochs, step, len(self.dataloader), loss.item())

            avg_loss = epoch_loss / len(self.dataloader)
            loss_history.append(avg_loss)
            logger.info("Epoch [{}/{}] completed. Average Loss: {:.4f}", epoch + 1, self.num_epochs, avg_loss)

        return loss_history

    def maybe_encode_image(self, data):
        if self.vae:
            with torch.no_grad():
                return self.vae.encode(data).latent_dist.sample() * self.vae.config.scaling_factor
        else:
            return data

    def maybe_get_conditional_input(self):
        return None

    def get_noisy_input(
        self,
        input: torch.Tensor,
        device: str = "cuda",
        normal_mean: float = 0.0,
        normal_std: float = 1.0,
        uniform_t: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = input.shape[0]
        noise = randn_tensor(input.shape, device=device)

        # Sample timestep from a log-normal distribution with mean 0 and std 1
        if uniform_t:
            t = torch.rand(b, device=device).float()
        else:
            t = torch.randn(b, device=device) * normal_std + normal_mean
            t = torch.sigmoid(t)

        t = t.clamp(0 + self.CLAMP_CONTINUOUS_TIME, 1 - self.CLAMP_CONTINUOUS_TIME)
        
        for _ in range(len(noise.shape) - 1):
            t = t.unsqueeze(1)

        noisy_input = (1 - t) * input + (self.MIN_STD + (1 - self.MIN_STD) * t) * noise
        return noise, noisy_input, t.squeeze()

    def rectified_flow_loss(
        self,
        input: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
        preds: torch.Tensor,
        use_weighting: bool = False,
        reduce: str = "mean",
    ) -> torch.Tensor:
        t = t.reshape(t.shape[0], *[1 for _ in range(len(input.shape) - len(t.shape))])

        target_flow = (1 - self.MIN_STD) * noise - input
        loss = F.mse_loss(preds.float(), target_flow.float(), reduction="none")
        
        if use_weighting:
            weight = self._logit_norm(t).detach()
            loss = loss * weight
        if reduce == "mean":
            loss = loss.mean()
        elif reduce == "none":
            loss = loss
        else:
            raise NotImplementedError

        return loss

    def discritize_timestep(self, t: Union[torch.Tensor, float], n_timesteps: int = 1000) -> torch.Tensor:
        return (t * n_timesteps).round().long()