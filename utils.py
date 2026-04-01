"""
utils.py

Utility functions for the atomic energy prediction project.

Functions:
- load_config: Load configuration from YAML file
- set_seed: Set random seeds for reproducibility
- check_cuda: Check GPU availability
- get_device: Get PyTorch device (GPU or CPU)
- create_loss_function: Create the loss function for training

Author: Aga
"""

import yaml
import torch
import numpy as np
import random
import os
from omegaconf import OmegaConf, DictConfig
from typing import Optional


def load_config(config_path: str = "config_atomic.yaml") -> DictConfig:
    """
    Load configuration from YAML file.
    
    Uses OmegaConf for flexible configuration management. This allows:
    - Dot notation access (config.model.hidden_layers)
    - Type checking
    - Easy merging of configs
    
    Args:
        config_path: Path to YAML configuration file
    
    Returns:
        Configuration object
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load YAML and convert to OmegaConf object
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    config = OmegaConf.create(config_dict)
    
    print(f"Loaded configuration from {config_path}")
    return config


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    This ensures that experiments can be reproduced exactly - the same seed
    will produce the same random numbers across runs. This affects:
    - NumPy random operations (data splitting, shuffling)
    - PyTorch random operations (weight initialization, dropout)
    - Python's random module
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # If using GPU, also set CUDA seeds
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        
        # Make CUDA operations deterministic (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"Set random seed to {seed}")


def check_cuda():
    """
    Check CUDA (GPU) availability and print information.
    
    GPUs dramatically speed up neural network training through parallel computation.
    This function checks if a GPU is available and prints useful information.
    """
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"CUDA available! Using GPU: {device_name}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"PyTorch version: {torch.__version__}")
    else:
        print("CUDA not available. Using CPU.")
        print("Note: Training will be slower on CPU. Consider using Google Colab for free GPU access.")


def get_device(config) -> torch.device:
    """
    Get the PyTorch device (GPU or CPU) for training.
    
    Args:
        config: Configuration object
    
    Returns:
        PyTorch device object
    """
    if config.general.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device('cpu')
        print(f"Using device: {device}")
    
    return device


def create_loss_function(criterion_name: str, reduction: str = 'none'):
    """
    Create the loss function for training.
    
    Loss functions measure how far the model's predictions are from the true values.
    During training, we minimize this loss.
    
    Common options for regression:
    - MSE (Mean Squared Error): Penalizes large errors heavily
    - MAE (Mean Absolute Error): More robust to outliers
    - Huber: Combination of MSE and MAE (smooth, robust)
    
    Args:
        criterion_name: Name of the loss function
    
    Returns:
        PyTorch loss function
    """
    if criterion_name == 'MSE':
        # Mean Squared Error: (prediction - target)²
        # Good for most regression tasks, penalizes large errors
        criterion = torch.nn.MSELoss(reduction=reduction)
    
    elif criterion_name == 'MAE':
        # Mean Absolute Error: |prediction - target|
        # More robust to outliers than MSE
        criterion = torch.nn.L1Loss(reduction=reduction)
    
    elif criterion_name == 'Huber':
        # Huber loss: MSE for small errors, MAE for large errors
        # Best of both worlds - smooth and robust
        criterion = torch.nn.SmoothL1Loss(reduction=reduction)
    
    else:
        raise ValueError(f"Unknown criterion: {criterion_name}")
    
    print(f"Using loss function: {criterion_name}")
    return criterion


def _get_elements_str(config) -> str:
    """Return a string identifying the element(s) in the config."""
    if hasattr(config.dataset, 'elements') and config.dataset.elements:
        return '_'.join(sorted(config.dataset.elements))
    elif hasattr(config.dataset, 'data_file') and config.dataset.data_file:
        return extract_element_from_filename(config.dataset.data_file)
    return ''


def get_experiment_tags(config) -> str:
    """
    Build a short tag string encoding target type and sample-weighting strategy.

    Target tag (transformations applied in order: binding → inverse → log):
        raw             – absolute E_level
        binded          – E_ion - E_level
        inv-raw         – A / E_level
        inv-binded      – A / (E_ion - E_level)
        log-raw         – log(E_level)
        log-binded      – log(E_ion - E_level)
        log-inv-raw     – log(A / E_level)
        log-inv-binded  – log(A / (E_ion - E_level))

    Weight tag:
        no-weights  – no sample weighting
        bins        – energy_bins strategy
        distance    – distance_to_ground strategy
        kde         – KDE strategy

    Returns:
        Tag string, e.g. 'log-binded_no-weights' or 'inv-binded_bins'
    """
    use_binding = config.dataset.get('use_binding_energy', False)
    use_inverse = config.dataset.get('use_inverse_target', False)
    if use_binding and use_inverse:
        target_tag = 'inv-binded'
    elif use_binding:
        target_tag = 'binded'
    elif use_inverse:
        target_tag = 'inv-raw'
    else:
        target_tag = 'raw'

    if config.dataset.get('use_log_target', False):
        target_tag = f'log-{target_tag}'

    use_weights = config.dataset.get('use_sample_weights', False)
    if use_weights:
        strategy = config.dataset.get('weight_strategy', 'energy_bins')
        weight_tag = {
            'energy_bins': 'bins',
            'distance_to_ground': 'distance',
            'kde': 'kde',
        }.get(strategy, strategy)
    else:
        weight_tag = 'no-weights'

    return f"{target_tag}_{weight_tag}"


def get_model_name_from_config(config) -> str:
    """
    Generate model checkpoint filename from configuration.

    Encodes element(s), optimizer, learning rate, batch size, architecture,
    dropout, target type, and sample-weighting strategy.

    Returns:
        Filename, e.g. 'best_model_K_Adam_lr0.001_bs16_128-64-32_drop0.3_binded_no_weights.pt'
    """
    elements_str = _get_elements_str(config)
    layers_str = '-'.join(str(h) for h in config.model.hidden_layers)
    tags = get_experiment_tags(config)
    return (
        f"best_model_{elements_str}"
        f"_{config.general.optimizer}"
        f"_lr{config.general.lr}"
        f"_bs{config.general.batch_size}"
        f"_{layers_str}"
        f"_drop{config.model.dropout}"
        f"_{tags}.pt"
    )


def get_predictions_filename(config) -> str:
    """
    Generate predictions CSV filename from configuration.

    Returns:
        Filename, e.g. 'predictions_K_binded_no_weights.csv'
    """
    elements_str = _get_elements_str(config)
    tags = get_experiment_tags(config)
    return f"predictions_{elements_str}_{tags}.csv"


def get_metrics_filename(config) -> str:
    """
    Generate metrics CSV filename from configuration.

    Returns:
        Filename, e.g. 'metrics_K_binded_no_weights.csv'
    """
    elements_str = _get_elements_str(config)
    tags = get_experiment_tags(config)
    return f"metrics_{elements_str}_{tags}.csv"


def extract_element_from_filename(filepath: str) -> str:
    """
    Extract element symbol from filename.

    Args:
        filepath: Path to data file

    Returns:
        Element symbol
    """
    import os

    filename = os.path.basename(filepath)
    name_without_ext = os.path.splitext(filename)[0]

    if '_' in name_without_ext:
        parts = name_without_ext.split('_')
        if len(parts) >= 3 and parts[0].lower() == 'energy':
            return parts[1]
        elif len(parts) >= 2:
            return parts[0]

    return name_without_ext


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, save_path):
    """
    Save a model checkpoint.
    
    Checkpoints allow you to:
    - Resume training if interrupted
    - Load the best model for evaluation
    - Track training progress over time
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch number
        train_loss: Training loss at this epoch
        val_loss: Validation loss at this epoch
        save_path: Where to save the checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint to {save_path}")


def load_checkpoint(model, optimizer, checkpoint_path, device):
    """
    Load a model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        checkpoint_path: Path to checkpoint file
        device: PyTorch device
    
    Returns:
        Tuple of (epoch, train_loss, val_loss)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    train_loss = checkpoint['train_loss']
    val_loss = checkpoint['val_loss']
    
    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"  Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    return epoch, train_loss, val_loss


def count_parameters(model):
    """
    Count trainable parameters in a PyTorch model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted string (e.g., "2h 15m 30s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"
