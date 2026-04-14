"""
run_experiments.py

Run a series of training + testing experiments sequentially, each with a
different set of config overrides. Results are appended (never overwritten)
to results/results_{element}.xlsx after every run.

Usage:
    python run_experiments.py
    python run_experiments.py --config my_config.yaml

To define experiments, edit the EXPERIMENTS list below.
Each entry is a dict of dotted config paths → override values.
Parameters not listed keep their value from config_atomic.yaml.
"""

import argparse
import os
import sys
import traceback

from omegaconf import OmegaConf

from utils import load_config, set_seed, check_cuda, get_model_name_from_config
from train_model import train_one_run
from test_model import test_one_run


# ── Define experiments here ──────────────────────────────────────────────────
# Each dict is one run. Only the keys you want to override are needed.
weights_strategies = ["energy_bins", "distance_to_ground", "kde"]  #
EXPERIMENTS = []

# [False, False, False], [True, False, False], [True, True, False], [True, False, True]
for bind, inv_target, log_target in [[False, False, False], [True, False, False], [True, False, True]]:
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


def run_single_experiment(config, idx: int, total: int) -> bool:
    """Train and test one experiment. Returns True on success."""
    print(f"\n{'=' * 60}")
    print(f"EXPERIMENT {idx + 1} / {total}")
    print(f"{'=' * 60}")

    set_seed(config.general.random_seed)

    # --- Training ---
    try:
        best_model_path, best_train_metrics, best_val_metrics = train_one_run(config)
        print("✓ Training completed")
    except Exception as exc:
        print(f"✗ Training failed: {exc}")
        traceback.print_exc()
        return False

    # --- Testing ---
    checkpoint = best_model_path or os.path.join(
        config.logging.save_dir, get_model_name_from_config(config)
    )

    if not os.path.exists(checkpoint):
        print(f"✗ Checkpoint not found: {checkpoint}")
        return False

    try:
        test_one_run(
            config, checkpoint,
            train_metrics=best_train_metrics,
            val_metrics=best_val_metrics,
        )
        print("✓ Testing completed")
    except Exception as exc:
        print(f"✗ Testing failed: {exc}")
        traceback.print_exc()
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run multiple training/testing experiments sequentially"
    )
    parser.add_argument(
        "--config", type=str, default="config_atomic.yaml",
        help="Base config file (default: config_atomic.yaml)"
    )
    args = parser.parse_args()

    total = len(EXPERIMENTS)
    print(f"\nRunning {total} experiment(s) sequentially")
    print(f"Base config : {args.config}")
    print(f"Results     : results/results_*.xlsx  (appended after each run)\n")

    summary = []

    for idx, overrides in enumerate(EXPERIMENTS):
        # Reload base config fresh for every experiment to avoid state leakage
        config = load_config(args.config)
        apply_overrides(config, overrides)

        print(f"\nOverrides for experiment {idx + 1}:")
        for k, v in overrides.items():
            print(f"  {k} = {v}")

        success = run_single_experiment(config, idx, total)
        summary.append((idx + 1, overrides, "OK" if success else "FAILED"))

    # --- Final summary ---
    print(f"\n{'=' * 60}")
    print("ALL EXPERIMENTS COMPLETED")
    print(f"{'=' * 60}")
    for exp_num, overrides, status in summary:
        tag = "  ".join(f"{k.split('.')[-1]}={v}" for k, v in overrides.items())
        print(f"  [{status}] Experiment {exp_num}: {tag}")
    print()

    failed = sum(1 for _, _, s in summary if s != "OK")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())