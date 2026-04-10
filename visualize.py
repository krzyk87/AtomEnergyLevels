"""
visualize.py

Visualization tools for analyzing atomic energy prediction results.

This module creates plots to help understand:
1. Model performance (predictions vs. true values)
2. Error distribution
3. Feature importance
4. Training progress

Author: Aga
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Tuple
import os
import json

# For ionization energy values and element extraction
try:
    from AtomicDataset import IONIZATION_ENERGIES
except Exception:
    IONIZATION_ENERGIES = {}
try:
    from utils import extract_element_from_filename
except Exception:
    def extract_element_from_filename(filepath: str) -> str:
        import os as _os
        name = _os.path.splitext(_os.path.basename(filepath))[0]
        if '_' in name:
            parts = name.split('_')
            if len(parts) >= 3 and parts[0].lower() == 'energy':
                return parts[1]
            elif len(parts) >= 2:
                return parts[0]
        return name


# Set nice plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
# Use a font that supports Unicode superscripts (e.g., cm⁻¹) to avoid glyph warnings on Windows
plt.rcParams['font.family'] = 'DejaVu Sans'


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


# ========================= New: Dataset distributions ========================= #
def _load_features_combine(features_csvs: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load one or multiple features CSVs and combine them, tagging Element per row.

    Returns combined DataFrame and the ordered list of elements encountered.
    """
    all_dfs = []
    elements_order: List[str] = []
    for csv_path in features_csvs:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Features CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)
        elem = 'Unknown'
        # Prefer explicit Element column if present
        if 'Element' in df.columns and pd.api.types.is_string_dtype(df['Element']):
            elem = str(df['Element'].iloc[0])
        else:
            elem = extract_element_from_filename(csv_path)
            df['Element'] = elem
        elements_order.append(elem)
        all_dfs.append(df)
    combined = pd.concat(all_dfs, ignore_index=True) if len(all_dfs) > 1 else all_dfs[0]
    return combined, elements_order


def _infer_split_path(features_csvs: List[str], default_dir: str = 'data') -> Optional[str]:
    """Infer split JSON path from features list and data dir."""
    elems = [extract_element_from_filename(p) for p in features_csvs]
    key = '_'.join(elems)
    candidate = os.path.join(default_dir, f"dataset_split_indices_{key}.json")
    if os.path.exists(candidate):
        return candidate
    # Try reversed order too
    if len(elems) > 1:
        key_rev = '_'.join(reversed(elems))
        candidate2 = os.path.join(default_dir, f"dataset_split_indices_{key_rev}.json")
        if os.path.exists(candidate2):
            return candidate2
    # Fallback: element-specific file when single element
    if len(elems) == 1:
        single = os.path.join(default_dir, f"dataset_split_indices_{elems[0]}.json")
        if os.path.exists(single):
            return single
    return None


def _compute_metrics(df: pd.DataFrame, big_A: float = 1000.0) -> Dict[str, np.ndarray]:
    """
    Compute metrics for plotting from the raw dataset rows.

    Metrics:
      - E_level: raw energy (cm^-1) from 'Level (cm-1)'
      - A_over_E: A / E_level
      - A_over_delta: A / (E_ion - E_level), where E_ion from IONIZATION_ENERGIES per element
    """
    if 'Level (cm-1)' not in df.columns:
        raise KeyError("Expected column 'Level (cm-1)' in features CSV")

    E = df['Level (cm-1)'].astype(float).to_numpy(copy=True)

    # Ionization per row using Element column if present
    if 'Element' in df.columns and len(IONIZATION_ENERGIES) > 0:
        elems = df['Element'].astype(str).to_numpy()
        # map each element to ionization energy (second tuple value)
        ion_map = {k: v[1] if isinstance(v, (list, tuple)) and len(v) >= 2 else v for k, v in IONIZATION_ENERGIES.items()}
        E_ion = np.array([ion_map.get(e, np.nan) for e in elems], dtype=float)
    else:
        # If unknown, set nan to avoid invalid division
        E_ion = np.full_like(E, np.nan, dtype=float)

    # Compute A / E_level with guard (only positive E)
    with np.errstate(divide='ignore', invalid='ignore'):
        A_over_E = big_A / E
        # Invalid or non-positive E should be set to nan
        A_over_E[~np.isfinite(A_over_E)] = np.nan
        A_over_E[E <= 0] = np.nan

        delta = E_ion - E
        A_over_delta = big_A / delta
        # Guard: require positive delta and finite values
        invalid = (~np.isfinite(A_over_delta)) | (delta <= 0)
        A_over_delta[invalid] = np.nan

    return {
        'E_level': E,
        'A_over_E': A_over_E,
        'A_over_delta': A_over_delta,
    }


def _subset_arrays(arr: np.ndarray, indices: List[int]) -> np.ndarray:
    idx = np.array(indices, dtype=int)
    if arr.shape[0] == 0 or idx.size == 0:
        return np.array([], dtype=arr.dtype)
    # Some split files may include indices beyond current df len when combining different subsets.
    valid = idx[(idx >= 0) & (idx < arr.shape[0])]
    return arr[valid]


def plot_dataset_energy_distributions(
    features_csvs: List[str],
    split_json: Optional[str] = None,
    big_A: float = 1000.0,
    output_dir: str = "visualizations/dataset_dists",
    bins: int = 60,
):
    """
    Create three plots showing distributions from the raw dataset:
      1) E_level (raw energy),
      2) A / E_level,
      3) A / (E_ion - E_level), using ionization energies if available.

    Each figure has three subplots for Train / Val / Test subsets using provided split indices.
    """
    os.makedirs(output_dir, exist_ok=True)

    df, elems = _load_features_combine(features_csvs)

    # Infer split path if not provided
    if split_json is None:
        inferred = _infer_split_path(features_csvs)
        if inferred is not None:
            split_json = inferred

    # Load split indices, or fallback to full as train if unavailable
    splits: Dict[str, List[int]] = {'train': list(range(len(df))), 'val': [], 'test': []}
    if split_json is not None and os.path.exists(split_json):
        try:
            with open(split_json, 'r', encoding='utf-8') as f:
                splits = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to read split JSON '{split_json}': {e}. Using all data as train.")
    else:
        if split_json is not None:
            print(f"Warning: Split JSON not found: {split_json}. Using all data as train.")

    # Compute metrics
    metrics = _compute_metrics(df, big_A=big_A)

    # Helper to clean data for plotting
    def clean(x: np.ndarray) -> np.ndarray:
        x = x[np.isfinite(x)]
        return x

    # Prepare subsets
    subsets = ['train', 'val', 'test']
    subset_data: Dict[str, Dict[str, np.ndarray]] = {}
    for sub in subsets:
        idxs = splits.get(sub, [])
        subset_data[sub] = {
            'E_level': clean(_subset_arrays(metrics['E_level'], idxs)),
            'A_over_E': clean(_subset_arrays(metrics['A_over_E'], idxs)),
            'A_over_delta': clean(_subset_arrays(metrics['A_over_delta'], idxs)),
        }

    # Plotting function
    def _plot_metric(metric_key: str, title: str, xlabel: str, filename: str):
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=False)
        for i, sub in enumerate(subsets):
            ax = axes[i]
            data = subset_data[sub][metric_key]
            if data.size == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.set_title(sub.capitalize())
                ax.set_xlabel(xlabel)
                ax.grid(True, alpha=0.3)
                continue
            # Choose bins adaptively within finite range
            try:
                ax.hist(data, bins=bins, edgecolor='black', alpha=0.7)
            except Exception:
                # In rare cases with extremely narrow ranges, fallback to auto bins
                ax.hist(data, bins='auto', edgecolor='black', alpha=0.7)
            ax.set_title(f"{sub.capitalize()} (n={data.size})")
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Count')
            ax.grid(True, alpha=0.3)
            # Basic stats box
            if data.size > 0:
                mean = float(np.mean(data))
                std = float(np.std(data))
                q50 = float(np.median(data))
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
                ax.text(0.97, 0.97, f"μ={mean:.3g}\nσ={std:.3g}\nmed={q50:.3g}",
                        transform=ax.transAxes, ha='right', va='top', bbox=props, fontsize=9)
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
        plt.close(fig)

    # 1) Raw E_level
    _plot_metric(
        'E_level',
        title='Distribution of Raw Energy Levels (E_level)',
        xlabel='E_level (cm⁻¹)',
        filename='dist_E_level.png'
    )

    # 2) A / E_level
    _plot_metric(
        'A_over_E',
        title=f'Distribution of A / E_level (A={big_A:g})',
        xlabel='A / E_level',
        filename='dist_A_over_E.png'
    )

    # 3) A / (E_ion - E_level)
    _plot_metric(
        'A_over_delta',
        title='Distribution of A / (E_ion - E_level)',
        xlabel='A / (E_ion - E_level)',
        filename='dist_A_over_delta.png'
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create visualization plots")
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML — auto-derives predictions filename when provided')
    parser.add_argument('--predictions', type=str,
                       default='saved_models/predictions_K_raw_no-weights.csv',
                       help='Path to predictions CSV file (ignored when --config is given)')
    parser.add_argument('--output_dir', type=str,
                       default='visualizations',
                       help='Directory to save plots')
    # New flags for dataset distributions
    parser.add_argument('--plot_dataset_dists', action='store_true',
                        help='If set, also plot dataset energy distributions per split')
    parser.add_argument('--features_csvs', type=str, default='data/K_features.csv',
                        help='Comma-separated list of features CSV paths (e.g., data/K_features.csv,data/Na_features.csv)')
    parser.add_argument('--split_json', type=str, default='',
                        help='Path to dataset split indices JSON (if omitted, will try to infer)')
    parser.add_argument('--big_A', type=float, default=100000.0,
                        help='Constant A used in A/E and A/(E_ion - E)')
    args = parser.parse_args()

    if args.config:
        from utils import load_config, get_predictions_filename
        _cfg = load_config(args.config)
        predictions_path = os.path.join(_cfg.logging.save_dir, get_predictions_filename(_cfg))
    else:
        predictions_path = args.predictions

    create_all_visualizations(predictions_path, args.output_dir)

    if args.plot_dataset_dists:
        features_list = [p.strip() for p in args.features_csvs.split(',') if p.strip()]
        split_path = args.split_json if args.split_json else None
        out_dir = os.path.join(args.output_dir, 'dataset_dists')
        plot_dataset_energy_distributions(
            features_csvs=features_list,
            split_json=split_path,
            big_A=args.big_A,
            output_dir=out_dir,
        )
