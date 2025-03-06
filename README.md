# Diffusion and Flow Matching Experimentation Framework

A flexible PyTorch implementation for experimenting with various diffusion and flow matching techniques in image generation.

## Overview

This repository provides a modular and extensible framework for implementing and testing different ideas in diffusion models and flow matching. The core architecture is based on SiT (Stochastic Interpolant Transformer), with a focus on making it easy to:

- Implement new model architectures
- Explore different training methodologies
- Experiment with various feature representation techniques
- Benchmark different approaches with standardized evaluation metrics

The repository includes several implemented approaches as examples:
- Base SiT implementation for flow matching
- Register token technique for enhanced representation
- REPA (Representation Alignment) for aligning with pre-trained features

## Key Features

- **Modular Design**: Easily extend with new models, trainers, and techniques
- **Configuration-Driven**: Hydra-based configuration system for experiment management
- **Multiple Models**: Various model sizes (S, B, L, XL) with different patch sizes (2, 4, 8)
- **Dataset Support**: Ready-to-use integration with CIFAR-10, CIFAR-100, ImageNet
- **Comprehensive Evaluation**: FID scoring, image generation, metric tracking
- **Optimized Training**: Checkpointing, resumable training, and efficient batch processing
- **Extensible Architecture**: Well-structured codebase designed for experimentation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/diffusion-flow-matching.git
cd diffusion-flow-matching
```

2. Create a conda environment and install dependencies:
```bash
conda create -n diffusion-experiments python=3.10
conda activate diffusion-experiments
pip install -r requirements.txt
```

## Project Structure

```
diffusion-flow-matching/
├── config/                 # Hydra configuration files
│   ├── dataset/            # Dataset configurations
│   ├── evaluate.yaml       # Evaluation configuration
│   └── sits2_*.yaml        # Model-specific configurations
├── data/                   # Data directory (created automatically)
├── src/                    # Source code
│   ├── dataset/            # Dataset implementations
│   ├── evaluator/          # Evaluation metrics and utilities
│   ├── model/              # Model implementations
│   │   ├── sit.py          # Stochastic Interpolant Transformer
│   │   └── projector_mlp.py # Projection modules
│   ├── trainer/            # Training implementations
│   │   ├── flow_matching_trainer.py      # Base flow matching trainer
│   │   └── flow_matching_repa_trainer.py # REPA training approach
│   └── utils/              # Utility functions
├── ckpts/                  # Model checkpoints (created during training)
├── evaluation_results/     # Evaluation results (created during evaluation)
├── train.py                # Main training script
└── evaluate.py             # Main evaluation script
```

## Usage

### Training

To train a model, use the `train.py` script with the appropriate configuration:

```bash
# Train base SiT model on CIFAR-100
python train.py --config-name sits2_cifar64

# Train with an experimental approach (register tokens)
python train.py --config-name sits2_cifar64_register

# Try the REPA approach
python train.py --config-name sits2_cifar64_repa

# Resume training from a checkpoint
python train.py --config-name sits2_cifar64 trainer.load_checkpoint_path=./ckpts/sits2_cifar64/checkpoint_epoch_10_step_1000.pth
```

### Evaluation

To evaluate trained models, use the `evaluate.py` script:

```bash
# Evaluate a model
python evaluate.py --config-name evaluate_sit2_cifar64

# Evaluate with different parameters
python evaluate.py --config-name evaluate_sit2_cifar64 evaluator.num_generated_images=1000
```

## Extending the Framework

### Adding a New Model

1. Create a new model class in `src/model/` that extends `BaseModel`
2. Implement the required methods, particularly `forward()`
3. Add configuration in the `config/` directory
4. Register model builder function if needed

Example:
```python
# src/model/my_new_model.py
from src.model import BaseModel
import torch.nn as nn

class MyNewDiffusionModel(BaseModel):
    def __init__(self, hidden_size, num_heads, **kwargs):
        super().__init__()
        # Initialize your model components
        
    def forward(self, x, t, y):
        # Implement the forward pass
        return output
```

### Implementing a New Training Method

1. Create a new trainer class in `src/trainer/` that extends `BaseTrainer`
2. Implement the `train()` method and any specialized logic
3. Add a corresponding configuration file

Example:
```python
# src/trainer/my_new_trainer.py
from src.trainer import BaseTrainer

class MyNewTrainer(BaseTrainer):
    def __init__(self, model, dataloader, **kwargs):
        super().__init__()
        # Initialize your trainer
        
    def train(self):
        # Implement your training logic
        return loss_history
```

### Creating a Custom Dataset Loader

1. Add your dataset class in `src/dataset/` extending `BaseDataset`
2. Implement the `get_dataloader()` method
3. Configure in the dataset configuration files

## Configuration System

The project uses Hydra for flexible configuration management. Key configurations:

- **Model configurations**: Define architecture, size, and special features
- **Training configurations**: Learning rates, optimizers, schedulers
- **Dataset configurations**: Data paths, preprocessing, batch sizes
- **Evaluation configurations**: Metrics, sample counts, output directories

Example of overriding configurations:
```bash
python train.py --config-name sits2_cifar64 model.params.patch_size=4 trainer.params.lr=5e-4
```

## Example Techniques

The repository includes several implemented techniques to serve as examples:

### Base SiT (Stochastic Interpolant Transformer)

The foundation model using transformer architecture for flow matching.

### Register Token Method

Adds learnable memory tokens to the transformer, enabling it to maintain information across the diffusion process.

```python
# Enable register tokens in your configuration:
model:
  params:
    use_register_tokens: true
    num_register_tokens: 2
```

### REPA (Representation Alignment)

Aligns intermediate features with a pre-trained vision model (DINO-v2) to improve generation quality.

```python
# Enable REPA in your configuration:
model:
  params:
    use_projector: true
    encoder_depth: 4
    z_dims: [1024]
trainer:
  _target_: src.trainer.flow_matching_repa_trainer.FlowMatchingREPATrainer
```

## Contributing

This framework is designed for experimentation, and contributions are highly encouraged! Here's how you can contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-new-idea`
3. Implement your experimental technique
4. Add tests if applicable
5. Commit your changes: `git commit -am 'Add new technique: XYZ'`
6. Push to the branch: `git push origin feature/my-new-idea`
7. Submit a pull request

### Contribution Guidelines

- Keep the modular design in mind - make it easy for others to build on your work
- Document your approach clearly with comments and docstrings
- Include example configuration files for your technique
- Share results or insights in the PR description

## License

[MIT License](LICENSE)

## Citation

If you use this framework in your research, please cite our work:

```bibtex
@misc{diffusion_flow_matching,
  author = {Your Name},
  title = {Diffusion and Flow Matching Experimentation Framework},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/diffusion-flow-matching}}
}
```

## Acknowledgments

- The SiT implementation is based on research on Stochastic Interpolant Transformers
- Some techniques leverage pre-trained vision models like DINO-v2
