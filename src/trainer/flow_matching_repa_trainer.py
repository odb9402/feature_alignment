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
    
    def __init__(self, model: BaseModel, dataloader, num_epochs=10, device='cpu', lr=1e-4, img_encoder="sd3"):
        super().__init__(model=model,
                         dataloader=dataloader,
                         num_epochs=num_epochs, 
                         device=device,
                         lr=lr,
                         img_encoder=img_encoder)
        self.dino_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
        self.dino_model = Dinov2Model.from_pretrained('facebook/dinov2-large')
        self.dino_model.eval()
        self.dino_model.requires_grad = False
        self.feature_align_loss_weight = 0.5

        logger.info("Initialized FlowMatchingREPATrainer with num_epochs: {}, device: {}, learning_rate: {}, image_encoder: {}",
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
                preds, features = self.model(noise_latents, timesteps, cond_input)
                denormed_image = (image + 1) / 2  # Scale from [-1, 1] to [0, 1]
                
                with torch.no_grad():
                    dino_input = self.dino_processor(images=denormed_image, do_rescale=False,
                                                     return_tensors='pt')
                    dino_features = self.dino_model(dino_input['pixel_values'].to(self.device))[0]

                # Rectified flow matching loss
                loss = self.rectified_flow_loss(latents, noise, t, preds, use_weighting=self.use_weighting)
                feature_align_loss = 0.0
                for feature in features:
                    feature_align_loss -= F.cosine_similarity(dino_features, feature).mean()
                
                total_loss = loss + self.feature_align_loss_weight * feature_align_loss
                total_loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                
                # Log each step's loss and relevant details
                logger.info("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Feature Align Loss: {:.4f}, Total Loss: {:.4f}",
                            epoch + 1, self.num_epochs, step, len(self.dataloader), loss.item(), feature_align_loss.item(), total_loss.item())

            avg_loss = epoch_loss / len(self.dataloader)
            loss_history.append(avg_loss)
            logger.info("Epoch [{}/{}] completed. Average Loss: {:.4f}", epoch + 1, self.num_epochs, avg_loss)

        return loss_history