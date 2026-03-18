"""
visualize.py

Visualization tools for analyzing atomic energy prediction results.

This module creates plots to help understand:
1. Model performance (predictions vs. true values)
2. Error distribution
3. Feature importance
4. Training progress

Author: Aga (ML Developer)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
import os


# Set nice plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11


def plot_predictions_vs_true(
    predictions: np.ndarray,
    targets: np.ndarray,
    save_path: str = "predictions_vs_true.png",
    title: str = "Predicted vs True Energy Levels"
):
    """
    Create scatter plot of predictions vs. true values.
    
    A perfect model would have all points on the diagonal line (y=x).
    Deviations from this line show prediction errors.
    
    Args:
        predictions: Model predictions (cm⁻¹)
        targets: True values (cm⁻¹)
        save_path: Where to save the plot
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Flatten arrays
    pred = predictions.flatten()
    true = targets.flatten()
    
    # Scatter plot
    ax.scatter(true, pred, alpha=0.5, s=50, edgecolors='black', linewidth=0.5)
    
    # Perfect prediction line (y = x)
    min_val = min(true.min(), pred.min())
    max_val = max(true.max(), pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 
            'r--', linewidth=2, label='Perfect prediction')
    
    # Labels and formatting
    ax.set_xlabel('True Energy Level (cm⁻¹)', fontsize=12)
    ax.set_ylabel('Predicted Energy Level (cm⁻¹)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add R² value
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    ss_res = np.sum((true - pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Add text box with metrics
    textstr = f'R² = {r2:.4f}\nMAE = {np.mean(np.abs(true - pred)):.2f} cm⁻¹'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    plt.close()


def plot_error_distribution(
    predictions: np.ndarray,
    targets: np.ndarray,
    save_path: str = "error_distribution.png"
):
    """
    Plot the distribution of prediction errors.
    
    This shows how errors are distributed. Ideally:
    - Centered around zero (no systematic bias)
    - Narrow distribution (small errors)
    - Symmetric (errors equally likely in both directions)
    
    Args:
        predictions: Model predictions (cm⁻¹)
        targets: True values (cm⁻¹)
        save_path: Where to save the plot
    """
    # Calculate errors
    errors = (predictions - targets).flatten()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram of errors
    axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero error')
    axes[0].set_xlabel('Prediction Error (cm⁻¹)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of Prediction Errors', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Add statistics
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    textstr = f'Mean: {mean_error:.2f} cm⁻¹\nStd: {std_error:.2f} cm⁻¹'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    axes[0].text(0.70, 0.95, textstr, transform=axes[0].transAxes, fontsize=11,
                verticalalignment='top', bbox=props)
    
    # Box plot of absolute errors
    abs_errors = np.abs(errors)
    axes[1].boxplot(abs_errors, vert=True)
    axes[1].set_ylabel('Absolute Error (cm⁻¹)', fontsize=12)
    axes[1].set_title('Absolute Error Distribution', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    median_abs_error = np.median(abs_errors)
    q75 = np.percentile(abs_errors, 75)
    q95 = np.percentile(abs_errors, 95)
    textstr = f'Median: {median_abs_error:.2f}\n75th %ile: {q75:.2f}\n95th %ile: {q95:.2f}'
    props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
    axes[1].text(0.60, 0.95, textstr, transform=axes[1].transAxes, fontsize=11,
                verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    plt.close()


def plot_error_vs_energy(
    predictions: np.ndarray,
    targets: np.ndarray,
    save_path: str = "error_vs_energy.png"
):
    """
    Plot prediction errors as a function of true energy level.
    
    This helps identify if the model performs differently for
    different energy ranges (e.g., better for low energies).
    
    Args:
        predictions: Model predictions (cm⁻¹)
        targets: True values (cm⁻¹)
        save_path: Where to save the plot
    """
    errors = (predictions - targets).flatten()
    true = targets.flatten()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scatter plot of errors vs. true values
    ax.scatter(true, errors, alpha=0.5, s=50, edgecolors='black', linewidth=0.5)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero error')
    
    ax.set_xlabel('True Energy Level (cm⁻¹)', fontsize=12)
    ax.set_ylabel('Prediction Error (cm⁻¹)', fontsize=12)
    ax.set_title('Prediction Error vs. True Energy Level', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    plt.close()


def plot_training_history(
    history_file: str = "training_history.csv",
    save_path: str = "training_history.png"
):
    """
    Plot training and validation loss over epochs.
    
    This shows how the model learned over time:
    - Training loss should decrease steadily
    - Validation loss should decrease, then plateau
    - Large gap = overfitting (model memorizing training data)
    
    Args:
        history_file: CSV file with training history
        save_path: Where to save the plot
    """
    if not os.path.exists(history_file):
        print(f"Training history file not found: {history_file}")
        return
    
    # Load training history
    df = pd.read_csv(history_file)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot losses
    ax.plot(df['epoch'], df['train_loss'], label='Training Loss', 
            marker='o', linewidth=2, markersize=4)
    ax.plot(df['epoch'], df['val_loss'], label='Validation Loss',
            marker='s', linewidth=2, markersize=4)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training History', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    plt.close()


def create_all_visualizations(
    predictions_file: str = "saved_models/predictions.csv",
    output_dir: str = "visualizations"
):
    """
    Create all visualization plots from predictions CSV.
    
    Args:
        predictions_file: CSV file with predictions
        output_dir: Directory to save plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load predictions
    if not os.path.exists(predictions_file):
        print(f"Predictions file not found: {predictions_file}")
        print("Please run testing first to generate predictions.")
        return
    
    print(f"\nCreating visualizations from {predictions_file}...")
    df = pd.read_csv(predictions_file)
    
    predictions = df['Predicted_Energy_cm-1'].values
    targets = df['True_Energy_cm-1'].values
    
    # Create plots
    plot_predictions_vs_true(
        predictions, targets,
        save_path=os.path.join(output_dir, "predictions_vs_true.png")
    )
    
    plot_error_distribution(
        predictions, targets,
        save_path=os.path.join(output_dir, "error_distribution.png")
    )
    
    plot_error_vs_energy(
        predictions, targets,
        save_path=os.path.join(output_dir, "error_vs_energy.png")
    )
    
    print(f"\nAll visualizations saved to {output_dir}/")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create visualization plots")
    parser.add_argument('--predictions', type=str, 
                       default='saved_models/predictions.csv',
                       help='Path to predictions CSV file')
    parser.add_argument('--output_dir', type=str,
                       default='visualizations',
                       help='Directory to save plots')
    args = parser.parse_args()
    
    create_all_visualizations(args.predictions, args.output_dir)
