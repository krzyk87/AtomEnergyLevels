"""
train_model.py

Training script for atomic energy level prediction.

This module handles:
1. Creating data loaders for train/validation sets
2. Initializing the model, optimizer, and loss function
3. Training loop with backpropagation
4. Validation and early stopping
5. Saving the best model

Key concepts for physics students:
- Epoch: One complete pass through the training data
- Batch: A subset of training samples processed together
- Forward pass: Computing predictions from inputs
- Backward pass: Computing gradients (how to adjust weights)
- Optimizer: Algorithm that updates weights based on gradients

Author: Aga
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple, Optional
import time
import os

from AtomicDataset import AtomicDataset
from AtomicModel import create_model
from utils import (
    create_loss_function,
    save_checkpoint,
    format_time, get_model_name_from_config
)


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    config
) -> float:
    """
    Train the model for one epoch.
    
    An epoch is one complete pass through the training data. During each epoch:
    1. Process batches of data
    2. Compute predictions (forward pass)
    3. Compute loss (how wrong are predictions)
    4. Compute gradients (backward pass)
    5. Update weights (optimization step)
    
    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer (e.g., Adam)
        device: CPU or GPU
        epoch: Current epoch number
        config: Configuration object
    
    Returns:
        Average training loss for this epoch
    """
    # Set model to training mode
    # This enables dropout and batch normalization training behavior
    model.train()
    
    running_loss = 0.0

    use_focal = config.training.get('use_focal_loss', False)
    focal_alpha = config.training.get('focal_loss_alpha', 0.5)
    use_sample_weights = (
            hasattr(train_loader.dataset, 'sample_weights')
            and train_loader.dataset.sample_weights is not None
    )
    
    # Iterate through batches of training data
    for batch_idx, (features, targets) in enumerate(train_loader):
        # Move data to GPU if available
        features = features.to(device)
        targets = targets.to(device)
        
        # Zero out gradients from previous iteration
        # PyTorch accumulates gradients, so we must clear them each time
        optimizer.zero_grad()
        
        # Forward pass: compute predictions
        predictions = model(features)
        
        # Compute loss (how far are predictions from true values)
        # Per-sample losses — shape: [batch_size, 1]
        loss = criterion(predictions, targets)

        # Start with uniform weights, shape: [batch_size, 1]
        weights = torch.ones_like(loss)

        # --- Sample weights (static, from energy distribution) ---
        if use_sample_weights:
            # Resolve which dataset samples this batch corresponds to.
            # train_loader.dataset.indices maps dataset positions to df row indices.
            # batch_idx * batch_size : (batch_idx+1) * batch_size gives the
            # position within the shuffled dataset for this batch.
            start = batch_idx * config.general.batch_size
            end = start + len(features)  # use len(features) not batch_size: last batch may be smaller
            batch_positions = list(range(start, min(end, len(train_loader.dataset))))

            sw = torch.FloatTensor([
                train_loader.dataset.sample_weights[pos]
                for pos in batch_positions
            ]).to(device).unsqueeze(1)  # [batch_size, 1]
            weights = weights * sw

        # --- Focal weights (dynamic, based on current prediction error) ---
        if use_focal:
            # Weight by error magnitude: harder samples get more weight
            # Alpha controls how aggressively to focus on hard samples
            with torch.no_grad():
                error_magnitude = torch.abs(predictions - targets).detach()
                focal_weights = (1 + error_magnitude) ** focal_alpha  # alpha ~ 0.5-2.0
                focal_weights = focal_weights / focal_weights.mean()  # normalize so average weight is 1.0
            weights = weights * focal_weights

        # Normalize combined weights so loss scale stays comparable across runs
        weights = weights / weights.mean()
        loss = (loss * weights).mean()

        # Apply sample weights if available
        # if hasattr(train_loader.dataset, 'sample_weights') and train_loader.dataset.sample_weights is not None:
        #     # Get batch indices
        #     batch_indices = train_loader.dataset.indices[
        #         batch_idx * config.general.batch_size:
        #         (batch_idx + 1) * config.general.batch_size
        #     ]
        #
        #     # Get weights for this batch
        #     batch_weights = torch.FloatTensor([
        #         train_loader.dataset.get_sample_weight(i)
        #         for i in range(len(features))
        #     ]).to(device)
        #
        #     # Apply weights to loss
        #     loss = (loss * batch_weights.unsqueeze(1)).mean()
        
        # Backward pass: compute gradients
        # This calculates how much each weight contributed to the loss
        loss.backward()
        
        # Gradient clipping: prevent exploding gradients
        # If gradients become too large, they can destabilize training
        if config.training.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                config.training.gradient_clip
            )
        
        # Optimization step: update weights based on gradients
        # The optimizer uses gradients to adjust weights to minimize loss
        optimizer.step()
        
        # Track loss for this batch
        running_loss += loss.item()
    
    # Average loss across all batches
    avg_loss = running_loss / len(train_loader)
    
    return avg_loss


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    val_dataset: AtomicDataset
) -> Tuple[float, float, float]:
    """
    Evaluate the model on validation data.
    
    Validation checks how well the model generalizes to unseen data.
    This is done without updating weights (no training).
    
    Args:
        model: Neural network model
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: CPU or GPU
        val_dataset: dataset object used for validation
    
    Returns:
        Tuple of (loss, MAE, RMSE)
        - loss: Average validation loss
        - MAE: Mean Absolute Error in cm⁻¹
        - RMSE: Root Mean Squared Error in cm⁻¹
    """
    # Set model to evaluation mode
    # This disables dropout and sets batch norm to evaluation mode
    model.eval()
    
    running_loss = 0.0
    all_predictions = []
    all_targets = []
    num_batches = 0
    
    # Disable gradient computation for efficiency
    # We're only evaluating, not training, so we don't need gradients
    with torch.no_grad():
        for features, targets in val_loader:
            features = features.to(device)
            targets = targets.to(device)
            
            # Forward pass only
            predictions = model(features)
            
            # Compute loss
            loss = criterion(predictions, targets)
            running_loss += loss.item()
            num_batches += 1
            
            # Store predictions and targets for metrics
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # Combine all batches
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    
    # Compute metrics
    avg_loss = running_loss / num_batches

    # Inverse transform BEFORE computing metrics
    predictions_cm = val_dataset.inverse_transform_target(all_predictions)
    targets_cm = val_dataset.inverse_transform_target(all_targets)

    mae = np.mean(np.abs(predictions_cm - targets_cm))
    rmse = np.sqrt(np.mean((predictions_cm - targets_cm) ** 2))
    
    return avg_loss, mae, rmse


def train_model(config, model, train_loader, val_loader, device, val_dataset):
    """
    Complete training pipeline.
    
    This function:
    1. Sets up optimizer and loss function
    2. Runs training loop for specified number of epochs
    3. Validates after each epoch
    4. Implements early stopping to prevent overfitting
    5. Saves the best model
    
    Args:
        config: Configuration object
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: CPU or GPU
        val_dataset: Dataset object used for validation
    
    Returns:
        Path to saved best model
    """
    # Create loss function
    criterion_train = create_loss_function(config.training.criterion, reduction='none')
    criterion_val = create_loss_function(config.training.criterion, reduction='mean')
    
    # Create optimizer
    # Optimizer adjusts model weights to minimize loss
    # Adam is a good default - it adapts learning rate automatically
    if config.general.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.general.lr,
            weight_decay=config.general.weight_decay
        )
    elif config.general.optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.general.lr,
            momentum=config.general.momentum,
            weight_decay=config.general.weight_decay
        )
    elif config.general.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=config.general.lr,
            weight_decay=config.general.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.general.optimizer}")
    
    # Learning rate scheduler (optional)
    # Reduces learning rate when validation loss stops improving
    # This helps fine-tune the model in later epochs
    scheduler = None
    if config.training.lr_scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.training.lr_scheduler_params.factor,
            patience=config.training.lr_scheduler_params.patience,
            min_lr=config.training.lr_scheduler_params.min_lr,
            verbose=True
        )
    
    # Early stopping: stop training if validation loss doesn't improve
    # This prevents overfitting (memorizing training data)
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Path for saving the best model
    save_dir = config.logging.save_dir
    os.makedirs(save_dir, exist_ok=True)

    model_filename = get_model_name_from_config(config)
    best_model_path = os.path.join(save_dir, model_filename)
    
    print(f"\n{'='*60}")
    print(f"Starting training for {config.general.epochs} epochs")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    # Training loop
    for epoch in range(1, config.general.epochs + 1):
        epoch_start = time.time()
        
        # Train for one epoch
        train_loss = train_one_epoch(
            model, train_loader, criterion_train, optimizer, device, epoch, config
        )
        
        # Validate
        val_loss, val_mae, val_rmse = validate(
            model, val_loader, criterion_val, device, val_dataset
        )
        
        # Update learning rate if using scheduler
        if scheduler is not None:
            scheduler.step(val_loss)
        
        epoch_time = time.time() - epoch_start
        
        # Print progress
        if epoch % config.logging.log_interval == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{config.general.epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val MAE: {val_mae:.2f} cm⁻¹ | "
                  f"Val RMSE: {val_rmse:.2f} cm⁻¹ | "
                  f"Time: {format_time(epoch_time)}")
        
        # Check if this is the best model so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save the best model
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss, best_model_path
            )
            print(f"  → New best model! Val Loss: {val_loss:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping check
        if patience_counter >= config.general.patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            print(f"Best validation loss: {best_val_loss:.4f}")
            break
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training completed in {format_time(total_time)}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {best_model_path}")
    print(f"{'='*60}\n")
    
    return best_model_path


def _extract_element_from_filename(filepath: str) -> str:
    """
    Extract element symbol from filename.

    Examples:
        'data/energy_Na_features.csv' → 'Na'
        'data/K_features.csv' → 'K'
    """
    import os
    filename = os.path.basename(filepath)

    # Try pattern: energy_ELEMENT_features.csv
    if '_' in filename:
        parts = filename.split('_')
        if len(parts) >= 2:
            # Remove 'energy' prefix if present
            if parts[0].lower() == 'energy':
                return parts[1]
            else:
                return parts[0]

    # Fallback: use filename without extension
    return os.path.splitext(filename)[0]


def train_one_run(config):
    """
    Run one complete training experiment.
    
    This is the main entry point for training. It:
    1. Creates datasets and data loaders
    2. Creates the model
    3. Trains the model
    4. Returns path to best model
    
    Args:
        config: Configuration object
    
    Returns:
        Path to saved best model
    """
    # Get device (GPU or CPU)
    if config.general.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = AtomicDataset(config, subset='train')
    
    # For val/test, use the same scalers fitted on training data
    val_dataset = AtomicDataset(
        config, 
        subset='val',
        scaler_features=train_dataset.scaler_features,
        scaler_target=train_dataset.scaler_target
    )
    
    # Create data loaders
    # DataLoader handles batching and shuffling
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.general.batch_size,
        shuffle=True,  # Shuffle training data each epoch
        num_workers=config.general.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.general.batch_size,
        shuffle=False,  # Don't shuffle validation data
        num_workers=config.general.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Create model
    print("\nCreating model...")
    input_dim = train_dataset.get_input_dim()
    model = create_model(config, input_dim)
    model = model.to(device)
    
    # Train model
    best_model_path = train_model(
        config, model, train_loader, val_loader, device, val_dataset
    )
    
    return best_model_path
