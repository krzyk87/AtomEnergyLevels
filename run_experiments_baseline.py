"""
run_experiments_baseline.py

Run a series of baseline ML experiments sequentially, each with a different
set of config overrides.  Results are appended (never overwritten) to
results/results_{element}.xlsx after every run — the same file used by the
DL pipeline — so DL and ML results can be compared directly.

Mirrors the structure of run_experiments.py but calls train_baseline_run /
test_baseline_run instead of the DL equivalents.

Usage:
    python run_experiments_baseline.py
    python run_experiments_baseline.py --model random_forest
    python run_experiments_baseline.py --config my_config.yaml
    python run_experiments_baseline.py --model gradient_boosting --config my_config.yaml

To define experiments, edit the EXPERIMENTS list below.
Each entry is a dict of dotted config paths → override values.
Parameters not listed keep their value from config_atomic.yaml.
"""

import argparse
import sys
import traceback

from omegaconf import OmegaConf

from utils import load_config, set_seed
from baseline_models import train_baseline_run, test_baseline_run


# ── Define experiments here ──────────────────────────────────────────────────
# Mirrors the experiment matrix in run_experiments.py:
#   3 target configurations × 4 weighting strategies = 12 experiments
weights_strategies = ["energy_bins", "distance_to_ground", "kde"]
EXPERIMENTS = []

for bind, inv_target, log_target in [
    [False, False, False],
    [True, False, False],
    [True, False, True],
]:
    EXPERIMENTS.append({
        "dataset.use_sample_weights": False,
        "dataset.use_binding_energy": bind,
        "dataset.use_inverse_target": inv_target,
        "dataset.use_log_target": log_target,
    })
    for strategy in weights_strategies:
        EXPERIMENTS.append({
            "dataset.use_sample_weights": True,
            "dataset.weight_strategy": strategy,
            "dataset.use_binding_energy": bind,
            "dataset.use_inverse_target": inv_target,
            "dataset.use_log_target": log_target,
        })
# ── End of experiment definitions ────────────────────────────────────────────


def apply_overrides(config, overrides: dict):
    """Apply a flat dict of dotted-path overrides to an OmegaConf config."""
    for key, value in overrides.items():
        OmegaConf.update(config, key, value, merge=True)
    return config


def run_single_baseline_experiment(config, idx: int, total: int) -> bool:
    """Train and test one baseline experiment. Returns True on success."""
    print(f"\n{'=' * 60}")
    print(f"BASELINE EXPERIMENT {idx + 1} / {total}")
    print(f"  Model: {config.model.baseline_model}")
    print(f"{'=' * 60}")

    set_seed(config.general.random_seed)

    # --- Training ---
    try:
        model_path, train_metrics, val_metrics = train_baseline_run(config)
        print("✓ Training completed")
    except Exception as exc:
        print(f"✗ Training failed: {exc}")
        traceback.print_exc()
        return False

    # --- Testing ---
    try:
        test_baseline_run(config, model_path, train_metrics=train_metrics, val_metrics=val_metrics)
        print("✓ Testing completed")
    except Exception as exc:
        print(f"✗ Testing failed: {exc}")
        traceback.print_exc()
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline ML experiments sequentially"
    )
    parser.add_argument(
        "--config", type=str, default="config_atomic.yaml",
        help="Base config file (default: config_atomic.yaml)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        choices=["linear_reg", "random_forest", "xgboost", "gradient_boosting"],
        help="Baseline model type (overrides config.model.baseline_model)",
    )
    args = parser.parse_args()

    total = len(EXPERIMENTS)
    print(f"\nRunning {total} baseline experiment(s) sequentially")
    print(f"Base config : {args.config}")
    if args.model:
        print(f"Model       : {args.model}  (--model override)")
    print(f"Results     : results/results_*.xlsx  (appended after each run)\n")

    summary = []

    for idx, overrides in enumerate(EXPERIMENTS):
        # Reload base config fresh for every experiment to avoid state leakage
        config = load_config(args.config)

        # Force baseline_ml architecture so config snapshot is accurate
        apply_overrides(config, {"model.architecture": "baseline_ml"})

        # Optional CLI override for model type
        if args.model:
            apply_overrides(config, {"model.baseline_model": args.model})

        apply_overrides(config, overrides)

        print(f"\nOverrides for experiment {idx + 1}:")
        print(f"  model.baseline_model = {config.model.baseline_model}")
        for k, v in overrides.items():
            print(f"  {k} = {v}")

        success = run_single_baseline_experiment(config, idx, total)
        summary.append((idx + 1, overrides, config.model.baseline_model, "OK" if success else "FAILED"))

    # --- Final summary ---
    print(f"\n{'=' * 60}")
    print("ALL BASELINE EXPERIMENTS COMPLETED")
    print(f"{'=' * 60}")
    for exp_num, overrides, model_type, status in summary:
        tag = "  ".join(f"{k.split('.')[-1]}={v}" for k, v in overrides.items())
        print(f"  [{status}] Experiment {exp_num} ({model_type}): {tag}")
    print()

    failed = sum(1 for _, _, _, s in summary if s != "OK")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
