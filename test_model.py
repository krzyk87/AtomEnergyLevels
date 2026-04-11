"""
test_model.py

Testing and evaluation script for atomic energy level prediction.

This module:
1. Loads a trained model
2. Evaluates it on the test set
3. Computes performance metrics
4. Saves predictions for analysis

Evaluation metrics:
- MSE: Mean Squared Error (average of squared errors)
- RMSE: Root Mean Squared Error (in original units: cm⁻¹)
- MAE: Mean Absolute Error (average absolute difference)
- R²: Coefficient of determination (0-1, higher is better)
- MAPE: Mean Absolute Percentage Error (percentage accuracy)

Author: Aga
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Dict, Any
import os

from AtomicDataset import AtomicDataset
from AtomicModel import create_model
from utils import load_checkpoint, get_model_name_from_config, get_predictions_filename, append_metrics_to_excel


def convert_predictions_to_absolute(
        binding_energy_predictions: np.ndarray,
        element: str
) -> np.ndarray:
    """
    Convert binding energy predictions back to absolute energy levels.

    E_level = E_ionization - ΔE_binding

    Args:
        binding_energy_predictions: Predicted binding energies (cm⁻¹)
        element: Element name (e.g., 'Na', 'K')

    Returns:
        Absolute energy levels (cm⁻¹)
    """
    from AtomicDataset import IONIZATION_ENERGIES

    if element not in IONIZATION_ENERGIES:
        raise ValueError(f"Unknown element: {element}")

    Z, E_ion = IONIZATION_ENERGIES[element]

    # E_level = E_ion - ΔE_binding
    energy_levels = E_ion - binding_energy_predictions

    return energy_levels


def compute_metrics(
    predictions: np.ndarray, 
    targets: np.ndarray
) -> Dict[str, float]:
    """
    Compute evaluation metrics for regression.
    
    Args:
        predictions: Model predictions (shape: [n_samples, 1])
        targets: True values (shape: [n_samples, 1])
    
    Returns:
        Dictionary of metrics
    """
    # Flatten arrays
    pred = predictions.flatten()
    true = targets.flatten()
    
    # Mean Squared Error: average of (prediction - target)²
    # Heavily penalizes large errors
    mse = np.mean((pred - true) ** 2)
    
    # Root Mean Squared Error: sqrt(MSE)
    # In the same units as the target (cm⁻¹)
    rmse = np.sqrt(mse)
    
    # Mean Absolute Error: average of |prediction - target|
    # More interpretable, less sensitive to outliers
    mae = np.mean(np.abs(pred - true))

    # Median Absolute Error: robust to outliers
    # Reports the 'typical' error, not skewed by worst-case predictions
    median_ae = float(np.median(np.abs(pred - true)))
    
    # R² (R-squared): proportion of variance explained
    # 1.0 = perfect predictions
    # 0.0 = predictions no better than mean
    # < 0 = predictions worse than mean
    ss_tot = np.sum((true - np.mean(true)) ** 2)  # Total variance
    ss_res = np.sum((true - pred) ** 2)            # Residual variance
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Mean Absolute Percentage Error: average of |error / target| * 100
    # Gives error as a percentage
    # Avoid division by zero for very small targets
    epsilon = 1e-8
    mape = np.mean(np.abs((true - pred) / (true + epsilon))) * 100
    
    # Maximum error: worst single prediction
    max_error = np.max(np.abs(pred - true))
    
    metrics = {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'median_ae': median_ae,  # add this
        'r2': float(r2),
        'mape': float(mape),
        'max_error': float(max_error)
    }
    
    return metrics


def test_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    test_dataset: AtomicDataset
) -> tuple:
    """
    Evaluate model on test set.
    
    Args:
        model: Trained neural network
        test_loader: DataLoader for test data
        device: CPU or GPU
        test_dataset: Test dataset (for inverse transform)
    
    Returns:
        Tuple of (metrics, predictions, targets)
    """
    # Set model to evaluation mode
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    print("Evaluating on test set...")
    
    # No gradient computation needed for evaluation
    with torch.no_grad():
        for features, targets in test_loader:
            features = features.to(device)
            targets = targets.to(device)
            
            # Get predictions
            predictions = model(features)
            
            # Store results
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # Combine all batches
    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)
    
    # Convert back to original scale (cm⁻¹)
    # The model was trained on normalized data, so we need to denormalize
    predictions_cm = test_dataset.inverse_transform_target(predictions)
    targets_cm = test_dataset.inverse_transform_target(targets)
    
    # Compute metrics on original scale
    metrics = compute_metrics(predictions_cm, targets_cm)
    
    return metrics, predictions_cm, targets_cm


def save_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    test_dataset: AtomicDataset,
    save_path: str = "predictions.csv"
):
    """
    Save predictions to CSV for analysis.
    
    This creates a file with:
    - True energy levels
    - Predicted energy levels
    - Absolute error
    - Percentage error
    - Feature values
    
    Args:
        predictions: Model predictions
        targets: True values
        test_dataset: Test dataset
        save_path: Where to save the CSV
    """
    # Get feature values for test samples
    feature_names = test_dataset.get_feature_names()
    X_test = test_dataset.df.loc[test_dataset.indices, feature_names]
    
    # Create results dataframe
    results = pd.DataFrame({
        'True_Energy_cm-1': targets.flatten(),
        'Predicted_Energy_cm-1': predictions.flatten(),
        'Absolute_Error_cm-1': np.abs(targets - predictions).flatten(),
        'Percentage_Error': (np.abs(targets - predictions) / (targets + 1e-8) * 100).flatten()
    })
    
    # Add feature columns
    for col in feature_names:
        results[col] = X_test[col].values
    
    # Add configuration and term info if available
    if 'Configuration' in test_dataset.df.columns:
        results['Configuration'] = test_dataset.df.loc[test_dataset.indices, 'Configuration'].values
    if 'Term' in test_dataset.df.columns:
        results['Term'] = test_dataset.df.loc[test_dataset.indices, 'Term'].values
    
    # Sort by absolute error (worst predictions first)
    results = results.sort_values('True_Energy_cm-1', ascending=True)
    
    # Save to CSV
    results.to_csv(save_path, index=False)
    print(f"\nSaved predictions to {save_path}")


def print_metrics(metrics: Dict[str, float]):
    """
    Print evaluation metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metric names and values
    """
    print("\n" + "="*60)
    print("TEST SET PERFORMANCE")
    print("="*60)
    print(f"Mean Squared Error (MSE):        {metrics['mse']:>12.2f}")
    print(f"Root Mean Squared Error (RMSE):  {metrics['rmse']:>12.2f} cm⁻¹")
    print(f"Mean Absolute Error (MAE):       {metrics['mae']:>12.2f} cm⁻¹")
    print(f"Median Absolute Error:           {metrics['median_ae']:>12.2f} cm⁻¹")
    print(f"R² Score:                        {metrics['r2']:>12.4f}")
    print(f"Mean Absolute % Error (MAPE):    {metrics['mape']:>12.2f}%")
    print(f"Maximum Error:                   {metrics['max_error']:>12.2f} cm⁻¹")
    print("="*60 + "\n")


def test_one_run(
    config,
    checkpoint_path: str = None,
    train_metrics: Dict[str, float] = None,
    val_metrics: Dict[str, float] = None,
) -> Dict[str, Any]:
    """
    Run evaluation on test set.
    
    Args:
        config: Configuration object
        checkpoint_path: Path to trained model checkpoint
    
    Returns:
        Dictionary of metrics
    """
    # Get device
    if config.general.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Load training dataset to get scalers
    print("\nLoading datasets...")
    train_dataset = AtomicDataset(config, subset='train')
    
    # Create test dataset with same scalers
    test_dataset = AtomicDataset(
        config,
        subset='test',
        scaler_features=train_dataset.scaler_features,
        scaler_target=train_dataset.scaler_target
    )
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.general.batch_size,
        shuffle=False,
        num_workers=config.general.num_workers
    )
    
    # Create model
    print("\nLoading model...")
    input_dim = test_dataset.get_input_dim()
    model = create_model(config, input_dim)
    model = model.to(device)
    
    # Load trained weights
    if checkpoint_path is None:
        # Auto-determine checkpoint path
        save_dir = config.logging.save_dir
        model_name = get_model_name_from_config(config)
        checkpoint_path = os.path.join(save_dir, model_name)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    optimizer = torch.optim.Adam(model.parameters())  # Dummy optimizer for loading
    load_checkpoint(model, optimizer, checkpoint_path, device)

    # Evaluate on test set.
    # test_model() calls test_dataset.inverse_transform_target() internally,
    # which handles ALL inverse steps in the correct order:
    #   1. Undo StandardScaler normalisation
    #   2. Undo A / E_target  (if use_inverse_target=True)
    # After this, predictions_cm and targets_cm are in the actual target space:
    #   - binding energy (cm⁻¹)  if use_binding_energy=True
    #   - absolute E_level (cm⁻¹) otherwise
    metrics, predictions_cm, targets_cm = test_model(
        model, test_loader, device, test_dataset
    )

    # If trained on binding energies, convert predictions back to absolute energy levels
    # This is a display-only step: E_level = E_ion - binding_energy.
    # We do NOT do this when use_inverse_target=True, because in that mode
    # the target is already A / binding_energy, and inverse_transform_target()
    # has already recovered the binding energy; the conversion below is
    # still correct and safe to apply.
    if config.dataset.get('use_binding_energy', False):
        # Get element from the dataset (supports both single and multi-element datasets)
        unique_elements = test_dataset.df['Element'].unique()
        if len(unique_elements) > 1:
            print(
                f"Warning: Multiple elements in test set: {unique_elements}. "
                f"Per-element conversion used."
            )
            # Multi-element: convert per row using the element column
            indices = test_dataset.indices
            elements = test_dataset.df.loc[indices, 'Element'].values
            abs_predictions = np.array([
                convert_predictions_to_absolute(
                    np.array([predictions_cm.flatten()[i]]), elem
                )[0]
                for i, elem in enumerate(elements)
            ])
            abs_targets = np.array([
                convert_predictions_to_absolute(
                    np.array([targets_cm.flatten()[i]]), elem
                )[0]
                for i, elem in enumerate(elements)
            ])
        else:
            element = unique_elements[0]
            abs_predictions = convert_predictions_to_absolute(predictions_cm, element)
            abs_targets = convert_predictions_to_absolute(targets_cm, element)

        predictions_cm = abs_predictions
        targets_cm = abs_targets
        metrics = compute_metrics(predictions_cm, targets_cm)

    # Print results
    print_metrics(metrics)

    features_config = {"n_features": test_dataset.get_input_dim(), "feature_names": test_dataset.get_feature_names()}

    # Append metrics + config settings to results Excel file
    append_metrics_to_excel(
        config, metrics,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        features=features_config
    )

    # Save predictions
    save_predictions(
        predictions_cm, targets_cm, test_dataset,
        save_path=os.path.join(config.logging.save_dir, get_predictions_filename(config))
    )
    
    return metrics


def _extract_element_from_filename(filepath: str) -> str:
    """Extract element symbol from filename (same as in train_model.py)."""
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


if __name__ == "__main__":
    import argparse
    from utils import load_config, set_seed, check_cuda
    
    parser = argparse.ArgumentParser(description="Test atomic energy prediction model")
    parser.add_argument('--config', type=str, default='config_atomic.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint')
    args = parser.parse_args()
    
    # Load config and setup
    config = load_config(args.config)
    set_seed(config.general.random_seed)
    check_cuda()
    
    # Run test
    metrics = test_one_run(config, args.checkpoint)
