"""
analyze_features.py

Permutation feature importance analysis for the trained atomic energy level MLP.

For each feature, the model is evaluated N times with that feature column randomly
shuffled. The mean increase in MAE compared to the unshuffled baseline gives an
estimate of how much the model relies on that feature.

Usage:
    python analyze_features.py [--config config_atomic.yaml] [--checkpoint path/to/model.pt]

Author: Aga
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from AtomicDataset import AtomicDataset
from AtomicModel import create_model
from utils import load_checkpoint, get_model_name_from_config, load_config, set_seed, check_cuda


# ---------------------------------------------------------------------------
# Feature group definitions for bar-chart colouring
# ---------------------------------------------------------------------------
_FEATURE_GROUPS = {
    'rydberg':  {'features': {'n_star', 'rydberg_pred', 'one_over_nstar_sq'},
                 'color': 'green',  'label': 'Rydberg'},
    'tm_ang':   {'features': {'J_sq', 'L_sq', 'S_sq', 'lande_so_term'},
                 'color': 'blue',   'label': 'TM angular momentum'},
    'd_elec':   {'features': {'n_3d', 'd_holes', 'd_from_half', 'is_half_filled'},
                 'color': 'orange', 'label': 'd-electron'},
    'screen':   {'features': {'Z_eff', 'Z_eff_sq'},
                 'color': 'red',    'label': 'Screening'},
    'so':       {'features': {'zeta_3d', 'E_so_estimate'},
                 'color': 'purple', 'label': 'Spin-orbit'},
}
_DEFAULT_COLOR = 'gray'


def _feature_color(name: str) -> str:
    """Return the bar colour for a feature based on its physics group."""
    for group in _FEATURE_GROUPS.values():
        if name in group['features']:
            return group['color']
    return _DEFAULT_COLOR


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _build_datasets(config):
    """
    Instantiate train and test AtomicDatasets using the same pipeline as test_model.py.

    The train dataset is loaded first so that its fitted scalers can be passed
    to the test dataset, ensuring consistent normalisation.

    Returns:
        (train_dataset, test_dataset)
    """
    train_dataset = AtomicDataset(config, subset='train')
    test_dataset = AtomicDataset(
        config,
        subset='test',
        scaler_features=train_dataset.scaler_features,
        scaler_target=train_dataset.scaler_target,
    )
    return train_dataset, test_dataset


def _load_model(config, test_dataset: AtomicDataset, checkpoint_path: str | None,
                device: torch.device) -> torch.nn.Module:
    """
    Load the trained MLP from a checkpoint file.

    If checkpoint_path is None the filename is derived from the config using
    the same naming convention as train_model.py.

    Args:
        config: OmegaConf config
        test_dataset: Used only to obtain the input dimension
        checkpoint_path: Explicit path or None to auto-derive
        device: Target device

    Returns:
        Loaded model in eval mode
    """
    if checkpoint_path is None:
        save_dir = config.logging.save_dir
        model_name = get_model_name_from_config(config)
        checkpoint_path = os.path.join(save_dir, model_name)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    input_dim = test_dataset.get_input_dim()
    model = create_model(config, input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters())   # dummy for load_checkpoint
    load_checkpoint(model, optimizer, checkpoint_path, device)
    model.eval()
    print(f"Loaded model from {checkpoint_path}")
    return model


def _get_test_arrays(test_dataset: AtomicDataset) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract normalised feature matrix and inverse-transformed target vector for the test set.

    Returns:
        X_test_norm : float32 array (n_samples, n_features) — already StandardScaled
        y_test_cm   : float64 array (n_samples,) — energy levels in cm⁻¹
    """
    X_test_norm = test_dataset.X.astype(np.float32)
    y_raw = test_dataset.y                                  # shape (n, 1), possibly scaled
    y_test_cm = test_dataset.inverse_transform_target(y_raw).flatten()
    return X_test_norm, y_test_cm


def _predict_cm(model: torch.nn.Module, X: np.ndarray,
                test_dataset: AtomicDataset, device: torch.device) -> np.ndarray:
    """
    Run a forward pass and return inverse-transformed predictions in cm⁻¹.

    Args:
        model: Trained MLP (eval mode expected)
        X: Feature array (n_samples, n_features), StandardScaled
        test_dataset: Provides inverse_transform_target
        device: Computation device

    Returns:
        Predictions in cm⁻¹, shape (n_samples,)
    """
    with torch.no_grad():
        tensor = torch.FloatTensor(X).to(device)
        raw = model(tensor).cpu().numpy()          # shape (n, 1), in model output space
    return test_dataset.inverse_transform_target(raw).flatten()


def _mae(pred: np.ndarray, true: np.ndarray) -> float:
    """Return mean absolute error."""
    return float(np.mean(np.abs(pred - true)))


# ---------------------------------------------------------------------------
# Permutation importance
# ---------------------------------------------------------------------------

def compute_permutation_importance(
    model: torch.nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_dataset: AtomicDataset,
    device: torch.device,
    n_repeats: int = 30,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Compute permutation feature importance on the test set.

    For each feature column, the column is randomly shuffled n_repeats times.
    The increase in MAE relative to the unshuffled baseline is recorded.

    Args:
        model: Trained MLP (eval mode)
        X_test: Normalised feature matrix (n_samples, n_features)
        y_test: Target values in cm⁻¹ (n_samples,)
        test_dataset: Provides feature names and inverse_transform_target
        device: Computation device
        n_repeats: Number of shuffle repetitions per feature
        random_seed: Base random seed for reproducibility

    Returns:
        DataFrame with columns [feature, mean_mae_increase, std_mae_increase]
        sorted by mean_mae_increase descending.
    """
    rng = np.random.default_rng(random_seed)
    feature_names = test_dataset.get_feature_names()
    n_features = len(feature_names)

    baseline_preds = _predict_cm(model, X_test, test_dataset, device)
    baseline_mae = _mae(baseline_preds, y_test)
    print(f"\nBaseline MAE on test set: {baseline_mae:.2f} cm⁻¹")

    records = []
    for i in tqdm(range(n_features), desc="Permutation importance", unit="feature"):
        mae_increases = []
        for r in range(n_repeats):
            X_perm = X_test.copy()
            X_perm[:, i] = rng.permutation(X_perm[:, i])
            perm_preds = _predict_cm(model, X_perm, test_dataset, device)
            perm_mae = _mae(perm_preds, y_test)
            mae_increases.append(perm_mae - baseline_mae)

        records.append({
            'feature': feature_names[i],
            'mean_mae_increase': float(np.mean(mae_increases)),
            'std_mae_increase': float(np.std(mae_increases)),
        })

    df = pd.DataFrame(records).sort_values('mean_mae_increase', ascending=False).reset_index(drop=True)
    df.insert(0, 'rank', range(1, len(df) + 1))
    return df, baseline_mae


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def print_importance_table(importance_df: pd.DataFrame):
    """Print a ranked table of feature importances to stdout."""
    total = importance_df['mean_mae_increase'].sum()
    total = total if total > 0 else 1.0   # avoid division by zero if all near-zero

    header = f"{'Rank':>4}  {'Feature':<28}  {'Mean MAE increase (cm⁻¹)':>24}  {'Std':>8}  {'Relative (%)':>12}"
    print("\n" + "=" * len(header))
    print("PERMUTATION FEATURE IMPORTANCE")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    positive_total = importance_df.loc[importance_df['mean_mae_increase'] > 0, 'mean_mae_increase'].sum()
    positive_total = positive_total if positive_total > 0 else 1.0

    for _, row in importance_df.iterrows():
        rel = row['mean_mae_increase'] / positive_total * 100 if row['mean_mae_increase'] > 0 else 0.0
        print(
            f"{int(row['rank']):>4}  {row['feature']:<28}  "
            f"{row['mean_mae_increase']:>24.2f}  "
            f"{row['std_mae_increase']:>8.2f}  "
            f"{rel:>12.1f}%"
        )
    print("=" * len(header))


def print_summary(importance_df: pd.DataFrame, baseline_mae: float):
    """Print a grouped summary of feature importances."""
    critical  = importance_df[importance_df['mean_mae_increase'] > 500]
    moderate  = importance_df[(importance_df['mean_mae_increase'] > 100) &
                               (importance_df['mean_mae_increase'] <= 500)]
    low       = importance_df[(importance_df['mean_mae_increase'] >= 0) &
                               (importance_df['mean_mae_increase'] < 50)]
    no_contrib = importance_df[importance_df['mean_mae_increase'] < 0]

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Baseline MAE: {baseline_mae:.2f} cm⁻¹\n")

    def _fmt(df, label):
        print(f"  {label} ({len(df)} features):")
        if df.empty:
            print("    (none)")
        else:
            for _, r in df.iterrows():
                print(f"    {r['feature']:<28}  {r['mean_mae_increase']:+.2f} ± {r['std_mae_increase']:.2f} cm⁻¹")

    _fmt(critical,   "CRITICAL  (MAE increase > 500 cm⁻¹)")
    _fmt(moderate,   "MODERATE  (100–500 cm⁻¹)")
    _fmt(low,        "LOW       (< 50 cm⁻¹ — candidates for removal)")
    _fmt(no_contrib, "NONE      (negative — safe to drop)")
    print("=" * 60)


def save_bar_chart(importance_df: pd.DataFrame, output_path: str, elements: list[str]):
    """
    Save a horizontal bar chart of permutation feature importances.

    Features are sorted with the most important at the top. Bars are coloured
    by physics group, error bars show ±1 std.

    Args:
        importance_df: DataFrame from compute_permutation_importance
        output_path: Full path to the output PNG file
        elements: List of element symbols (for the chart title)
    """
    # Sort ascending so top bar appears at the top of the figure
    df = importance_df.sort_values('mean_mae_increase', ascending=True)

    colors = [_feature_color(f) for f in df['feature']]

    fig_height = max(6, len(df) * 0.35)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    y_pos = np.arange(len(df))
    ax.barh(y_pos, df['mean_mae_increase'], xerr=df['std_mae_increase'],
            color=colors, ecolor='black', capsize=3, height=0.7)

    ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['feature'], fontsize=9)
    ax.set_xlabel('Mean MAE increase (cm⁻¹)', fontsize=11)

    elements_str = ' '.join(elements)
    ax.set_title(f'Permutation Feature Importance — {elements_str} Test Set', fontsize=12)

    # Legend for feature groups
    from matplotlib.patches import Patch
    legend_handles = [Patch(color=g['color'], label=g['label']) for g in _FEATURE_GROUPS.values()]
    legend_handles.append(Patch(color=_DEFAULT_COLOR, label='Other / base'))
    ax.legend(handles=legend_handles, loc='lower right', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved bar chart to {output_path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_permutation_importance(config, checkpoint_path: str | None = None):
    """
    Orchestrate the full permutation importance analysis pipeline.

    Loads the trained model and test set, runs permutation importance,
    prints a ranked table and summary, and saves the bar chart.

    Args:
        config: OmegaConf config (must include dataset, model, logging sections)
        checkpoint_path: Path to the .pt model file, or None to auto-derive
    """
    # --- Device ---
    device = torch.device(
        'cuda' if config.general.device == 'cuda' and torch.cuda.is_available() else 'cpu'
    )
    print(f"Using device: {device}")

    # --- Permutation importance config ---
    pi_cfg = config.get('permutation_importance', {})
    n_repeats   = int(pi_cfg.get('n_repeats',   30))
    random_seed = int(pi_cfg.get('random_seed', 42))
    output_dir  = pi_cfg.get('output_dir', None)

    # --- Datasets ---
    print("\nLoading datasets...")
    train_dataset, test_dataset = _build_datasets(config)

    # --- Model ---
    model = _load_model(config, test_dataset, checkpoint_path, device)

    # --- Test arrays ---
    X_test, y_test = _get_test_arrays(test_dataset)
    print(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")

    # --- Permutation importance ---
    importance_df, baseline_mae = compute_permutation_importance(
        model, X_test, y_test, test_dataset, device,
        n_repeats=n_repeats, random_seed=random_seed,
    )

    # --- Determine output directory ---
    if output_dir is None:
        if checkpoint_path is not None:
            output_dir = os.path.dirname(os.path.abspath(checkpoint_path))
        else:
            output_dir = config.logging.save_dir
    os.makedirs(output_dir, exist_ok=True)

    # --- Print results ---
    print_importance_table(importance_df)
    print_summary(importance_df, baseline_mae)

    # --- Save table ---
    csv_path = os.path.join(output_dir, 'feature_importance.csv')
    importance_df.to_csv(csv_path, index=False)
    print(f"\nSaved importance table to {csv_path}")

    # --- Save chart ---
    elements = list(config.dataset.elements) if hasattr(config.dataset, 'elements') else ['?']
    chart_path = os.path.join(output_dir, 'feature_importance.png')
    save_bar_chart(importance_df, chart_path, elements)

    return importance_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Permutation feature importance for atomic energy MLP")
    parser.add_argument('--config',     type=str, default='config_atomic.yaml',
                        help='Path to config YAML file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (.pt file)')
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.general.random_seed)
    check_cuda()

    run_permutation_importance(cfg, args.checkpoint)
