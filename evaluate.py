#!/usr/bin/env python3
import os
import glob
import hydra
import torch
import logging
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from loguru import logger
from src.experiment import GenericExperimentFactory
from src.evaluator import ModelEvaluator
from src.utils import set_all_seeds

@hydra.main(config_path='config', config_name='evaluate')
def run_evaluation(cfg: DictConfig):
    # Set seed for reproducibility
    set_all_seeds(cfg.seed)
    
    # Configure logging
    log_file = "evaluation.log"
    logger.add(log_file, rotation="1 MB", retention="10 days", level="DEBUG")
    
    # Save the configuration for reproducibility
    with open("evaluation_config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    
    logger.info(f"Evaluation Configuration:\n{OmegaConf.to_yaml(cfg, resolve=True)}")
    
    # Initialize experiment factory
    factory = GenericExperimentFactory()
    
    # Create model
    logger.info("Creating model...")
    try:
        model = factory.create_model(cfg)
        logger.info(f"Model created: {model.__class__.__name__}")
    except Exception as e:
        logger.error(f"Failed to create model: {str(e)}")
        return
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    
    # If validation dataset config is provided, create validation dataloader
    val_dataloader = None
    if 'validation_dataset' in cfg:
        try:
            val_dataset_cfg = OmegaConf.create(cfg.validation_dataset)
            val_dataloader = factory.create_dataloader(val_dataset_cfg)
            logger.info(f"Validation dataloader created with batch size: {val_dataset_cfg.dataset.batch_size}")
        except Exception as e:
            logger.error(f"Failed to create validation dataloader: {str(e)}")
    
    # Create training dataloader (might be needed for FID evaluation)
    try:
        train_dataloader = factory.create_dataloader(cfg)
        logger.info(f"Training dataloader created with batch size: {cfg.dataset.batch_size}")
    except Exception as e:
        logger.error(f"Failed to create training dataloader: {str(e)}")
        if val_dataloader is None:
            logger.error("Both training and validation dataloaders failed to initialize. Exiting...")
            return
        train_dataloader = None
    
    # Determine device
    device = cfg.evaluator.device if 'device' in cfg.evaluator else 'cuda:0'
    if device.startswith('cuda') and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    # Create evaluator
    try:
        evaluator = ModelEvaluator(
            model=model,
            dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            device=device,
            seed=cfg.seed,
            channel=cfg.evaluator.get('channel', 3),
            input_size=cfg.evaluator.get('input_size', 64),
            n_timesteps=cfg.evaluator.get('n_timesteps', 1000),
            img_encoder=cfg.evaluator.get('img_encoder', None),
            output_dir=cfg.evaluator.get('output_dir', './evaluation_results'),
            save_images=cfg.evaluator.get('save_images', True),
            n_classes=cfg.evaluator.get('n_classes', 100)
        )
    except Exception as e:
        logger.error(f"Failed to create evaluator: {str(e)}")
        return
    
    # Prepare the real images directory if needed for FID calculation
    real_images_dir = cfg.evaluator.get('real_images_dir', None)
    if real_images_dir is not None and cfg.evaluator.get('calculate_fid', False):
        logger.info(f"Preparing real images directory: {real_images_dir}")
        try:
            real_images_dir = evaluator.prepare_real_images_dir(
                real_images_dir, 
                num_images=cfg.evaluator.get('num_generated_images', 500)
            )
        except Exception as e:
            logger.error(f"Failed to prepare real images directory: {str(e)}")
            if cfg.evaluator.get('calculate_fid', False):
                logger.warning("FID calculation will be skipped due to real images directory preparation failure")
    
    # Checkpoints to evaluate
    checkpoint_paths = []
    
    # Multiple specific checkpoints
    if 'checkpoints' in cfg.evaluator:
        checkpoint_paths = cfg.evaluator.checkpoints
        
    # Or checkpoint directory with pattern
    elif 'checkpoint_dir' in cfg.evaluator:
        checkpoint_dir = Path(cfg.evaluator.checkpoint_dir)
        checkpoint_pattern = cfg.evaluator.get('checkpoint_pattern', '*.pth')
        checkpoint_paths = list(glob.glob(str(checkpoint_dir / checkpoint_pattern)))
        
        # Sort checkpoints if possible
        checkpoint_paths.sort()
        
        # Optionally limit number of checkpoints
        if 'max_checkpoints' in cfg.evaluator and len(checkpoint_paths) > cfg.evaluator.max_checkpoints:
            # Evenly sample checkpoints
            step = len(checkpoint_paths) // cfg.evaluator.max_checkpoints
            checkpoint_paths = checkpoint_paths[::step][:cfg.evaluator.max_checkpoints]
    
    # Single checkpoint
    elif 'checkpoint_path' in cfg.evaluator:
        checkpoint_paths = [cfg.evaluator.checkpoint_path]
    
    if not checkpoint_paths:
        logger.error("No checkpoints found for evaluation")
        return
        
    logger.info(f"Found {len(checkpoint_paths)} checkpoints to evaluate")
    
    # Run evaluation
    if len(checkpoint_paths) > 1:
        # Multiple checkpoints - run comparative evaluation
        try:
            metrics = evaluator.evaluate_checkpoints(
                checkpoint_paths=checkpoint_paths,
                real_images_dir=real_images_dir,
                calculate_val_loss=cfg.evaluator.get('calculate_val_loss', True),
                calculate_fid_score=cfg.evaluator.get('calculate_fid', False),
                plot_metrics=cfg.evaluator.get('plot_metrics', True),
                num_val_batches=cfg.evaluator.get('num_val_batches', None),
                num_generated_images=cfg.evaluator.get('num_generated_images', 100)
            )
            
            logger.info(f"Completed evaluation of {len(checkpoint_paths)} checkpoints")
            logger.info(f"Results saved to {cfg.evaluator.get('output_dir', './evaluation_results')}")
        except Exception as e:
            logger.error(f"Failed during checkpoint evaluation: {str(e)}")
            
    elif len(checkpoint_paths) == 1:
        # Single checkpoint evaluation
        try:
            metrics = evaluator.evaluate_checkpoint(
                checkpoint_path=checkpoint_paths[0],
                real_images_dir=real_images_dir,
                calculate_val_loss=cfg.evaluator.get('calculate_val_loss', True),
                calculate_fid_score=cfg.evaluator.get('calculate_fid', False),
                num_val_batches=cfg.evaluator.get('num_val_batches', None),
                num_generated_images=cfg.evaluator.get('num_generated_images', 100)
            )
            
            logger.info(f"Completed evaluation of checkpoint: {checkpoint_paths[0]}")
            logger.info(f"Metrics: {metrics}")
        except Exception as e:
            logger.error(f"Failed during checkpoint evaluation: {str(e)}")

if __name__ == "__main__":
    run_evaluation()