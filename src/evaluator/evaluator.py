import os
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from loguru import logger
from torchvision.utils import save_image, make_grid
from cleanfid import fid

from ..model import BaseModel
from ..utils import set_all_seeds


class ModelEvaluator:
    """
    Evaluator for SiT models to compute various metrics and save evaluation results.
    """
    def __init__(
        self,
        model: BaseModel,
        dataloader: torch.utils.data.DataLoader = None,
        val_dataloader: torch.utils.data.DataLoader = None,
        device: str = 'cuda:0',
        seed: int = 42,
        channel: int = 3,
        input_size: int = 64,
        n_timesteps: int = 1000,
        img_encoder: str = None,
        output_dir: str = './evaluation_results',
        save_images: bool = True,
        n_classes: int = 100
    ):
        """
        Initialize the evaluator with model and evaluation settings.
        
        Args:
            model: The model to evaluate
            dataloader: Dataloader for generating samples (can be training data)
            val_dataloader: Dataloader for validation loss calculation
            device: Device to run evaluation on
            seed: Random seed for reproducibility
            channel: Number of image channels
            input_size: Image size (assumed square)
            n_timesteps: Number of timesteps for flow matching
            img_encoder: Image encoder type, if any
            output_dir: Directory to save evaluation results
            save_images: Whether to save generated images
            n_classes: Number of classes (if class conditional generation)
        """
        self.model = model.to(device)
        self.model.eval()
        
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.seed = seed
        self.channel = channel
        self.input_size = input_size
        self.n_timesteps = n_timesteps
        self.img_encoder = img_encoder
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_images = save_images
        self.n_classes = n_classes

        # For VAE encoding/decoding if needed
        self.vae = None
        if img_encoder == "sd3":
            from diffusers.models import AutoencoderKL
            self.vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", subfolder="vae")
            self.vae = self.vae.to(device)
            self.vae.eval()
        
        # Metrics storage
        self.metrics_history = {}
        
        logger.info(f"Initialized ModelEvaluator with seed: {seed}, device: {device}, input_size: {input_size}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model weights from checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.eval()
        logger.info(f"Loaded checkpoint from {checkpoint_path}")

    def calculate_validation_loss(self, n_batches: Optional[int] = None, metric_file: Optional[str] = None) -> float:
        """
        Calculate validation loss on validation dataset.
        
        Args:
            n_batches: Number of batches to use for validation (None for all)
            metric_file: Path to existing metric file to check for cached validation loss
            
        Returns:
            Average validation loss
        """
        # Check if validation loss already exists in metrics file
        if metric_file is not None and os.path.exists(metric_file):
            try:
                with open(metric_file, 'r') as f:
                    metrics = json.load(f)
                if 'validation_loss' in metrics and not np.isnan(metrics['validation_loss']):
                    logger.info(f"Found existing validation loss: {metrics['validation_loss']:.6f}")
                    return metrics['validation_loss']
            except Exception as e:
                logger.warning(f"Failed to load existing validation loss: {str(e)}")
        
        if self.val_dataloader is None:
            logger.warning("No validation dataloader provided, skipping validation loss")
            return float('nan')
        
        set_all_seeds(self.seed)  # Ensure deterministic evaluation
        total_loss = 0.0
        batch_count = 0
        
        with torch.no_grad():
            for idx, (images, labels) in enumerate(tqdm(self.val_dataloader, desc="Calculating validation loss")):
                if n_batches is not None and idx >= n_batches:
                    break
                    
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Encode images if using VAE
                latents = self._maybe_encode_image(images)
                
                # Sample timesteps
                noise, noisy_latents, t = self._get_noisy_input(latents)
                timesteps = self._discretize_timestep(t, self.n_timesteps)
                
                # Get model predictions
                if hasattr(self.model, 'use_projector') and self.model.use_projector:
                    preds, _ = self.model(noisy_latents, timesteps, labels)
                else:
                    preds = self.model(noisy_latents, timesteps, labels)
                
                # Calculate loss (same as in trainer)
                loss = self._rectified_flow_loss(latents, noise, t, preds)
                total_loss += loss.item()
                batch_count += 1
        
        avg_loss = total_loss / batch_count if batch_count > 0 else float('nan')
        logger.info(f"Validation Loss: {avg_loss:.6f}")
        return avg_loss

    def generate_images(
        self, 
        num_images: int = 50, 
        batch_size: int = 10, 
        nfe: int = 50, 
        target_dir: Optional[str] = None,
        class_labels: Optional[torch.Tensor] = None,
        skip_if_exists: bool = True
    ) -> List[torch.Tensor]:
        """
        Generate images using the model.
        
        Args:
            num_images: Number of images to generate
            batch_size: Batch size for generation
            nfe: Number of function evaluations (sampling steps)
            target_dir: Directory to save generated images
            class_labels: Optional tensor of class labels for conditional generation
            skip_if_exists: Skip generation if target directory already contains the expected number of images
            
        Returns:
            List of generated image tensors
        """
        set_all_seeds(self.seed)  # Ensure deterministic generation
        
        if target_dir is None:
            target_dir = self.output_dir / "generated_images"
        
        target_path = Path(target_dir)
        target_path.mkdir(exist_ok=True, parents=True)
        
        # Check if images already exist
        if skip_if_exists:
            existing_images = list(target_path.glob("sample_*.png"))
            if len(existing_images) >= num_images:
                logger.info(f"Found {len(existing_images)} existing images in {target_dir}, skipping generation")
                # Return empty list as we don't load the images
                return []
            
        # Prepare for sampling
        self.model.eval()
        generated_images = []
        
        # Determine number of batches
        num_batches = (num_images + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc="Generating images"):
                # Adjust batch size for the last batch if needed
                curr_batch_size = min(batch_size, num_images - batch_idx * batch_size)
                
                # Generate random noise
                noise = torch.randn(curr_batch_size, self.channel, self.input_size, self.input_size).to(self.device)
                
                # Get class labels (either provided or random)
                if class_labels is not None:
                    batch_labels = class_labels[batch_idx * batch_size:batch_idx * batch_size + curr_batch_size]
                else:
                    batch_labels = torch.randint(0, self.n_classes, (curr_batch_size,)).to(self.device)
                
                # Sample using Euler method for flow matching
                x = noise.clone()
                
                # Uniform time steps from 1 to 0
                time_steps = torch.linspace(1.0, 0.0, nfe + 1)[:-1].to(self.device)
                
                for i, t in enumerate(time_steps):
                    # Broadcast t to match batch size
                    timestep = torch.ones(curr_batch_size, device=self.device) * t
                    timestep = self._discretize_timestep(timestep, self.n_timesteps)
                    
                    # Predict velocity field
                    if hasattr(self.model, 'use_projector') and self.model.use_projector:
                        v, _ = self.model(x, timestep, batch_labels)
                    else:
                        v = self.model(x, timestep, batch_labels)
                    
                    # Euler step
                    dt = 1.0 / nfe
                    x = x - dt * v
                
                # Decode if using VAE
                images = self._maybe_decode_latents(x)
                
                # Clamp to valid image range
                images = torch.clamp(images, 0, 1)
                
                # Save individual images
                if self.save_images:
                    for j in range(curr_batch_size):
                        img_idx = batch_idx * batch_size + j
                        save_image(images[j], f"{target_dir}/sample_{img_idx:05d}.png")
                
                # Add to list of generated images
                generated_images.append(images.cpu())
        
        # Save grid of images
        if self.save_images and generated_images:
            all_images = torch.cat(generated_images, dim=0)
            grid_size = min(8, int(np.sqrt(len(all_images))))
            grid = make_grid(all_images[:grid_size**2], nrow=grid_size)
            save_image(grid, f"{target_dir}/grid.png")
            
        logger.info(f"Generated {len(torch.cat(generated_images, dim=0)) if generated_images else 0} images at {target_dir}")
        return generated_images

    def calculate_fid(
        self, 
        real_images_dir: str,
        generated_images_dir: Optional[str] = None,
        num_images: int = 500,
        batch_size: int = 50,
        metric_file: Optional[str] = None
    ) -> float:
        """
        Calculate Fréchet Inception Distance (FID) between real and generated images.
        
        Args:
            real_images_dir: Directory containing real images (will be populated if empty)
            generated_images_dir: Directory for generated images (will generate if None)
            num_images: Number of images to generate/use
            batch_size: Batch size for generation
            metric_file: Path to existing metric file to check for cached FID score
            
        Returns:
            FID score
        """
        # Check if FID score already exists in metrics file
        if metric_file is not None and os.path.exists(metric_file):
            try:
                with open(metric_file, 'r') as f:
                    metrics = json.load(f)
                if 'fid_score' in metrics and not np.isnan(metrics['fid_score']):
                    logger.info(f"Found existing FID score: {metrics['fid_score']:.4f}")
                    return metrics['fid_score']
            except Exception as e:
                logger.warning(f"Failed to load existing FID score: {str(e)}")
        
        # First ensure real images directory is properly prepared
        try:
            real_dir = self.prepare_real_images_dir(real_images_dir, num_images, batch_size)
        except Exception as e:
            logger.error(f"Failed to prepare real images directory: {str(e)}")
            return float('nan')
        
        # Generate images for FID calculation if needed
        if generated_images_dir is None:
            generated_images_dir = str(self.output_dir / "fid_samples")
            try:
                # Generate images
                self.generate_images(
                    num_images=num_images, 
                    batch_size=batch_size,
                    target_dir=generated_images_dir,
                    skip_if_exists=True
                )
            except Exception as e:
                logger.error(f"Failed to generate images for FID calculation: {str(e)}")
                return float('nan')
        
        # Check if there are enough images in the generated directory
        gen_dir = Path(generated_images_dir)
        gen_images = list(gen_dir.glob("*.png"))
        
        if len(gen_images) < num_images * 0.9:  # Allow for some flexibility (90% of expected images)
            logger.warning(f"Not enough generated images found in {generated_images_dir}. Generating more...")
            try:
                self.generate_images(
                    num_images=num_images, 
                    batch_size=batch_size,
                    target_dir=generated_images_dir,
                    skip_if_exists=False
                )
            except Exception as e:
                logger.error(f"Failed to generate additional images: {str(e)}")
                return float('nan')
        
        # Calculate FID using clean-fid
        try:
            fid_score = fid.compute_fid(real_dir, generated_images_dir)
            logger.info(f"FID Score: {fid_score:.4f}")
            return fid_score
        except Exception as e:
            logger.error(f"Failed to compute FID score: {str(e)}")
            return float('nan')

    def evaluate_checkpoint(
        self, 
        checkpoint_path: str,
        real_images_dir: Optional[str] = None,
        calculate_val_loss: bool = True,
        calculate_fid_score: bool = False,
        num_val_batches: Optional[int] = None,
        num_generated_images: int = 100,
        skip_if_metrics_exist: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate a single checkpoint with multiple metrics.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            real_images_dir: Directory with real images for FID calculation
            calculate_val_loss: Whether to calculate validation loss
            calculate_fid_score: Whether to calculate FID score
            num_val_batches: Number of validation batches to use
            num_generated_images: Number of images to generate
            skip_if_metrics_exist: Skip evaluation if metrics file already exists
            
        Returns:
            Dictionary of metrics
        """
        # Extract checkpoint name for saving results
        checkpoint_name = os.path.basename(checkpoint_path)
        
        # Create directory for this checkpoint's results
        checkpoint_dir = self.output_dir / f"checkpoint_{checkpoint_name}"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Path to metrics file
        metrics_file = checkpoint_dir / "metrics.json"
        
        # Check if metrics already exist
        if skip_if_metrics_exist and metrics_file.exists():
            try:
                with open(metrics_file, 'r') as f:
                    existing_metrics = json.load(f)
                
                # Check if all required metrics exist
                required_metrics = []
                if calculate_val_loss:
                    required_metrics.append('validation_loss')
                if calculate_fid_score:
                    required_metrics.append('fid_score')
                
                has_all_metrics = all(metric in existing_metrics for metric in required_metrics)
                
                if has_all_metrics:
                    logger.info(f"Found existing metrics for {checkpoint_name}, skipping evaluation")
                    self.metrics_history[checkpoint_name] = existing_metrics
                    return existing_metrics
            except Exception as e:
                logger.warning(f"Failed to load existing metrics: {str(e)}")
        
        # Load checkpoint
        try:
            self.load_checkpoint(checkpoint_path)
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {str(e)}")
            return {'error': f"Failed to load checkpoint: {str(e)}"}
        
        metrics = {}
        
        # Calculate validation loss
        if calculate_val_loss:
            try:
                val_loss = self.calculate_validation_loss(
                    n_batches=num_val_batches,
                    metric_file=metrics_file if metrics_file.exists() else None
                )
                metrics['validation_loss'] = val_loss
            except Exception as e:
                logger.error(f"Failed to calculate validation loss: {str(e)}")
                metrics['validation_loss'] = float('nan')
        
        # Generate images and save in checkpoint directory
        generated_dir = checkpoint_dir / "generated_images"
        try:
            self.generate_images(
                num_images=num_generated_images,
                target_dir=str(generated_dir),
                batch_size=num_val_batches,  # Use smaller of num_val_batches or 50
                skip_if_exists=skip_if_metrics_exist
            )
        except Exception as e:
            logger.error(f"Failed to generate images: {str(e)}")
            metrics['generation_error'] = str(e)
        
        # Calculate FID if requested
        if calculate_fid_score and real_images_dir is not None:
            try:
                fid_score = self.calculate_fid(
                    real_images_dir=real_images_dir,
                    generated_images_dir=str(generated_dir),
                    num_images=num_generated_images,
                    metric_file=metrics_file if metrics_file.exists() else None
                )
                metrics['fid_score'] = fid_score
            except Exception as e:
                logger.error(f"Failed to calculate FID score: {str(e)}")
                metrics['fid_score'] = float('nan')
        
        # Save metrics
        self.metrics_history[checkpoint_name] = metrics
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        logger.info(f"Evaluation complete for {checkpoint_name}")
        return metrics

    def evaluate_checkpoints(
        self, 
        checkpoint_paths: List[str],
        real_images_dir: Optional[str] = None,
        calculate_val_loss: bool = True,
        calculate_fid_score: bool = False,
        plot_metrics: bool = True,
        num_generated_images: int = 100,
        num_val_batches: Optional[int] = 64,
        skip_if_metrics_exist: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate multiple checkpoints and plot evolution of metrics.
        
        Args:
            checkpoint_paths: List of paths to checkpoint files
            real_images_dir: Directory with real images for FID calculation
            calculate_val_loss: Whether to calculate validation loss
            calculate_fid_score: Whether to calculate FID score
            plot_metrics: Whether to plot metrics evolution
            skip_if_metrics_exist: Skip evaluation for checkpoints with existing metrics
            
        Returns:
            Dictionary of metrics for all checkpoints
        """
        all_metrics = {}
        
        # Check if all metrics file exists
        all_metrics_file = self.output_dir / "all_metrics.json"
        if skip_if_metrics_exist and all_metrics_file.exists():
            try:
                with open(all_metrics_file, 'r') as f:
                    all_metrics = json.load(f)
                
                # Check if all checkpoints are present in the metrics file
                checkpoint_names = [os.path.basename(cp) for cp in checkpoint_paths]
                has_all_checkpoints = all(f"checkpoint_{name}" in all_metrics for name in checkpoint_names)
                
                if has_all_checkpoints:
                    logger.info("Found metrics for all checkpoints, skipping evaluation")
                    
                    # Still plot metrics if requested
                    if plot_metrics:
                        self._plot_metrics_evolution(all_metrics)
                    
                    return all_metrics
            except Exception as e:
                logger.warning(f"Failed to load existing metrics: {str(e)}")
        
        # Evaluate each checkpoint
        for checkpoint_path in checkpoint_paths:
            checkpoint_name = os.path.basename(checkpoint_path)
            logger.info(f"Evaluating checkpoint: {checkpoint_name}")
            
            metrics = self.evaluate_checkpoint(
                checkpoint_path=checkpoint_path,
                real_images_dir=real_images_dir,
                calculate_val_loss=calculate_val_loss,
                calculate_fid_score=calculate_fid_score,
                num_generated_images=num_generated_images,
                num_val_batches=num_val_batches,
                skip_if_metrics_exist=skip_if_metrics_exist
            )
            
            all_metrics[f"checkpoint_{checkpoint_name}"] = metrics
        
        # Save all metrics to a combined file
        with open(all_metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=4)
        
        # Plot metrics evolution if requested
        if plot_metrics:
            self._plot_metrics_evolution(all_metrics)
        
        return all_metrics

    def prepare_real_images_dir(
        self,
        real_images_dir: str,
        num_images: int = 500,
        batch_size: int = 50,
    ) -> str:
        """
        Prepare the real images directory for FID calculation.
        If the directory is empty, save images from validation dataloader.
        
        Args:
            real_images_dir: Directory to store real images
            num_images: Number of real images to save
            batch_size: Batch size for processing validation data
            
        Returns:
            Path to the prepared real images directory
        """
        real_dir = Path(real_images_dir)
        real_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if directory is empty
        image_files = list(real_dir.glob('*.png')) + list(real_dir.glob('*.jpg'))
        
        if len(image_files) > 0:
            logger.info(f"Found {len(image_files)} existing real images in {real_images_dir}")
            return str(real_dir)
        
        # Directory is empty, try to save images from validation dataloader
        logger.info(f"Real images directory {real_images_dir} is empty. Attempting to save images from validation dataloader.")
        
        if self.val_dataloader is None:
            # Try training dataloader if validation is not available
            if self.dataloader is None:
                error_msg = "Cannot save real images: both validation and training dataloaders are None."
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            logger.warning("Validation dataloader not available. Using training dataloader instead.")
            dataloader = self.dataloader
        else:
            dataloader = self.val_dataloader
        
        try:
            images_saved = 0
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(dataloader, desc="Saving real images")):
                    # Handle different dataloader formats
                    if isinstance(batch, (list, tuple)):
                        images = batch[0]  # Assuming first element is images
                    else:
                        images = batch
                    
                    # Convert to device if needed
                    images = images.to(self.device)
                    
                    # Save individual images
                    for i in range(images.shape[0]):
                        if images_saved >= num_images:
                            break
                        
                        img_path = real_dir / f"real_image_{images_saved:05d}.png"
                        save_image(images[i].cpu(), img_path)
                        images_saved += 1
                    
                    if images_saved >= num_images:
                        break
            
            logger.info(f"Successfully saved {images_saved} real images to {real_images_dir}")
            
            if images_saved == 0:
                error_msg = "Failed to save any real images from dataloader."
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            return str(real_dir)
            
        except Exception as e:
            error_msg = f"Failed to save real images: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _plot_metrics_evolution(self, metrics: Dict[str, Dict[str, float]]) -> None:
        """
        Plot the evolution of metrics across checkpoints.
        
        Args:
            metrics: Dictionary of metrics for each checkpoint
        """
        # Extract checkpoint names and order them
        checkpoint_names = list(metrics.keys())
        
        # Try to extract epoch/step numbers for proper ordering
        def extract_number(name):
            # Extract numbers from strings like "checkpoint_epoch_10_step_5000.pth"
            parts = name.split('_')
            for i, part in enumerate(parts):
                if part == "epoch" and i+1 < len(parts):
                    return int(parts[i+1])
                if part == "step" and i+1 < len(parts):
                    return int(parts[i+1])
            return 0
        
        # Sort checkpoints by epoch/step if possible
        try:
            checkpoint_names.sort(key=extract_number)
        except:
            # Fall back to alphabetical sorting
            checkpoint_names.sort()
        
        # Get available metrics
        all_metric_names = set()
        for checkpoint_metrics in metrics.values():
            all_metric_names.update(checkpoint_metrics.keys())
        
        # Plot each metric
        for metric_name in all_metric_names:
            # Skip error messages and non-numeric metrics
            if metric_name.endswith('_error') or not any(
                isinstance(metrics.get(checkpoint_name, {}).get(metric_name, None), (int, float))
                for checkpoint_name in checkpoint_names
            ):
                continue
                
            plt.figure(figsize=(10, 6))
            
            # Extract metric values for each checkpoint
            x_values = []
            y_values = []
            
            for checkpoint_name in checkpoint_names:
                if metric_name in metrics.get(checkpoint_name, {}) and not np.isnan(metrics[checkpoint_name][metric_name]):
                    x_values.append(extract_number(checkpoint_name))
                    y_values.append(metrics[checkpoint_name][metric_name])
            
            # Skip if no data
            if not x_values:
                continue
                
            plt.plot(x_values, y_values, 'o-', linewidth=2)
            plt.xlabel('Training Step/Epoch')
            plt.ylabel(metric_name.replace('_', ' ').title())
            plt.title(f'Evolution of {metric_name.replace("_", " ").title()} Across Checkpoints')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Save plot
            plt.tight_layout()
            plt.savefig(self.output_dir / f"{metric_name}_evolution.png", dpi=300)
            plt.close()
            
        logger.info(f"Saved metric evolution plots to {self.output_dir}")

    def _maybe_encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images with VAE if available."""
        if self.vae is None:
            return images
            
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
            latents = (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        return latents

    def _maybe_decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents with VAE if available."""
        if self.vae is None:
            return latents
            
        with torch.no_grad():
            latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
            images = self.vae.decode(latents).sample
        return images

    def _get_noisy_input(
        self, 
        input: torch.Tensor,
        normal_mean: float = 0.0,
        normal_std: float = 1.0,
        uniform_t: bool = False,
        min_std: float = 0.0,
        clamp_time: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get noisy input for flow matching."""
        b = input.shape[0]
        noise = torch.randn(input.shape, device=self.device)

        # Sample timestep
        if uniform_t:
            t = torch.rand(b, device=self.device).float()
        else:
            t = torch.randn(b, device=self.device) * normal_std + normal_mean
            t = torch.sigmoid(t)

        t = t.clamp(0 + clamp_time, 1 - clamp_time)

        # Add dimensions to match input shape
        for _ in range(len(noise.shape) - 1):
            t = t.unsqueeze(1)

        # Apply noise
        noisy_input = (1 - t) * input + (min_std + (1 - min_std) * t) * noise
        return noise, noisy_input, t.squeeze()

    def _rectified_flow_loss(
        self,
        input: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
        preds: torch.Tensor,
        use_weighting: bool = False,
        reduce: str = "mean",
        min_std: float = 0.0
    ) -> torch.Tensor:
        """Calculate rectified flow loss."""
        t = t.reshape(t.shape[0], *[1 for _ in range(len(input.