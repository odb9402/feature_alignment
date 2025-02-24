import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from typing import Tuple, Union
from loguru import logger  # Added loguru import

from diffusers.utils.torch_utils import randn_tensor
from diffusers.models import AutoencoderKL

from . import BaseTrainer
from ..model import BaseModel

class FlowMatchingTrainer(BaseTrainer):

    MIN_STD: float = 0.0  # minimum size of std for the flow matching
    CLAMP_CONTINUOUS_TIME: float = 0.0

    def __init__(self,
                 model: BaseModel,
                 dataloader,
                 num_epochs=10,
                 device='cpu',
                 lr=1e-4,
                 img_encoder="sd3",
                 n_timesteps=1000,
                 save_checkpoint_dir=None,
                 save_checkpoint_interval=10000):
        self.n_timesteps = n_timesteps
        self.model = model.to(device)
        self.dataloader = dataloader
        self.num_epochs = num_epochs
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.use_weighting = False
        self.current_epoch = 0
        self.current_step = 0
        self.total_step = 0

        self.save_checkpoint_dir = save_checkpoint_dir
        os.makedirs(self.save_checkpoint_dir, exist_ok=True)
        self.save_checkpoint_interval = save_checkpoint_interval

        if img_encoder == "sd3":
            self.vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", subfolder="vae")
        else:
            self.vae = None

        logger.info("Initialized FlowMatchingTrainer with num_epochs: {}, device: {}, learning_rate: {}, image_encoder: {}",
                    num_epochs, device, lr, img_encoder)

    def train(self):
        self.model.train()
        loss_history = []

        start_epoch = self.current_epoch
        start_step = self.current_step
    
        for epoch in range(start_epoch, self.num_epochs):
            epoch_loss = 0.0
            logger.info("Starting epoch {}/{}", epoch + 1, self.num_epochs)
            self.current_epoch = epoch # update current epoch at the beginning of each epoch

            for step_idx, (image, label) in enumerate(self.dataloader, 1):
                if epoch == start_epoch and step_idx <= start_step:
                    continue # Skip steps already done if loading from checkpoint
                self.current_step = step_idx # update current step at the beginning of each step

                self.optimizer.zero_grad()

                image = image.to(self.device)
                latents = self.maybe_encode_image(image)
                cond_input = self.maybe_get_conditional_input(label)
                cond_input = cond_input.to(self.device)

                noise, noise_latents, t = self.get_noisy_input(latents)

                timesteps = self.discritize_timestep(t, self.n_timesteps)

                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    preds = self.model(noise_latents, timesteps, cond_input)
                    loss = self.rectified_flow_loss(latents, noise, t, preds, use_weighting=self.use_weighting)

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                
                # Log each step's loss and relevant details
                logger.info(f"Epoch: {epoch + 1}/{self.num_epochs}, "
                            f"Step: {step_idx}/{len(self.dataloader)}, "
                            f"Total Step: {self.total_step}, "
                            f"lr: {self.optimizer.param_groups[0]['lr']:.4f}, "
                            f"Loss: {loss.item():.6f}, ")

                
                self.total_step += 1
                # Save checkpoint every `save_checkpoint_interval` steps (adjust as needed)
                if self.total_step % self.save_checkpoint_interval == 0:
                    checkpoint_path = f"checkpoint_epoch_{epoch+1}_step_{self.total_step}.pth"
                    self.save_checkpoint(f"{self.save_checkpoint_dir}/{checkpoint_path}")
                    logger.info(f"Checkpoint saved to {self.save_checkpoint_dir}/{checkpoint_path}")


            avg_loss = epoch_loss / len(self.dataloader)
            loss_history.append(avg_loss)
            logger.info("Epoch [{}/{}] completed. Average Loss: {:.4f}", epoch + 1, self.num_epochs, avg_loss)
            self.current_step = 0 # reset step count after each epoch

        return loss_history

    def maybe_encode_image(self, data):
        """
        If self.vae is defined, this function encodes the given data using the
        pre-trained VAE. Otherwise, it simply returns the input data.

        Args:
            data: The input data to be encoded.

        Returns:
            The encoded input data.
        """
        if self.vae:
            with torch.no_grad():
                return (self.vae.encode(data).latent_dist.sample() - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        else:
            return data

    def maybe_get_conditional_input(self, label):
        return label

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
        """
        Computes the rectified flow loss (RFL) for the given predictions and inputs.

        Args:
            input: The input tensor, shape (B, C, H, W)
            noise: The noise tensor, shape (B, C, H, W)
            t: The timestep tensor, shape (B,) or (B, 1)
            preds: The predicted tensor, shape (B, C, H, W)
            use_weighting: Whether to use the logit-normalized timestep as a weight
            reduce: How to reduce the loss, either "mean" or "none". If "none", the loss is returned as a tensor of shape (B, C, H, W)

        Returns:
            The rectified flow loss, a scalar if reduce is "mean", otherwise a tensor of shape (B, C, H, W)
        """
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
        return (t * n_timesteps).round()#.long()

    def save_checkpoint(self, filepath):
        """
        Saves the current state of the trainer to a checkpoint file.

        Args:
            filepath: The path to save the checkpoint to.
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'step': self.current_step,
            'total_step': self.total_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint to {filepath}")

    def load_checkpoint(self, filepath):
        """
        Loads the trainer state from a checkpoint file.

        Args:
            filepath: The path to the checkpoint file.

        Returns:
            epoch: The epoch number loaded from checkpoint.
            step: The step number loaded from checkpoint.
        """
        if not os.path.exists(filepath):
            logger.warning(f"Checkpoint file not found: {filepath}. Starting from scratch.")
            return 0, 0  # Return default values if checkpoint not found

        checkpoint = torch.load(filepath, map_location=self.device)
        self.current_epoch = checkpoint['epoch']
        self.current_step = checkpoint['step']
        self.total_step = checkpoint['total_step']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Loaded checkpoint from {filepath}, Epoch: {self.current_epoch}, Step: {self.current_step}")
        return self.current_epoch, self.current_step