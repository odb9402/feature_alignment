import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Union
from loguru import logger  # Added loguru import

from transformers.models.dinov2.modeling_dinov2 import Dinov2Model
from transformers import AutoImageProcessor

from . import BaseTrainer
from .flow_matching_trainer import FlowMatchingTrainer
from ..model import BaseModel

class FlowMatchingREPATrainer(FlowMatchingTrainer):
    
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
        super().__init__(model=model,
                         dataloader=dataloader,
                         num_epochs=num_epochs, 
                         device=device,
                         lr=lr,
                         img_encoder=img_encoder,
                         n_timesteps=n_timesteps,
                         save_checkpoint_dir=save_checkpoint_dir,
                         save_checkpoint_interval=save_checkpoint_interval)
        self.dino_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
        self.dino_model = Dinov2Model.from_pretrained('facebook/dinov2-large')
        self.dino_model.eval()
        self.dino_model.requires_grad = False
        self.dino_model = self.dino_model.to(dtype=torch.bfloat16, device=self.device)
        self.feature_align_loss_weight = 0.5

        logger.info("Initialized FlowMatchingREPATrainer with num_epochs: {}, device: {}, learning_rate: {}, image_encoder: {}",
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
                dino_input = self.dino_processor(image, return_tensors="pt").pixel_values.to(self.device)
                cond_input = self.maybe_get_conditional_input(label)
                cond_input = cond_input.to(self.device)

                noise, noise_latents, t = self.get_noisy_input(latents)

                timesteps = self.discritize_timestep(t, self.n_timesteps)
                
                # Get image encoder (dino) features to align with flow matching features
                with torch.no_grad():
                    dino_features = self.dino_model(dino_input.to(dtype=torch.bfloat16))[0]
                    dino_features = dino_features[:, 1:] # remove CLS token, [B, 256, C]
                    dino_features = dino_features.to(torch.float32)
                
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    preds, dit_intermediate_features = self.model(noise_latents, timesteps, cond_input)
                    
                    loss = self.rectified_flow_loss(latents, noise, t, preds, use_weighting=self.use_weighting)
                    feature_align_loss = 0.0

                    for dit_feature in dit_intermediate_features:
                        # might consider several features for alignment
                        feature_align_loss = - F.cosine_similarity(dino_features, dit_feature).mean()
                    total_loss = loss + self.feature_align_loss_weight * feature_align_loss
                    self.optimizer.step()

                total_loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                
                # Log each step's loss and relevant details
                logger.info(f"Epoch: {epoch + 1}/{self.num_epochs}, "
                            f"Step: {step_idx}/{len(self.dataloader)}, "
                            f"Total Step: {self.total_step}, "
                            f"lr: {self.optimizer.param_groups[0]['lr']:.4f}, "
                            f"Loss: {loss.item():.6f}, " 
                            f"Feature Align Loss: {feature_align_loss.item():.6f}")

                
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
    