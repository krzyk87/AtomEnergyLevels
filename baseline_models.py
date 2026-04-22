"""
baseline_models.py

Scikit-learn baseline ML models for atomic energy level prediction.

Supports the same data pipeline, feature engineering, and results format
as the DL approach so experiments are directly comparable.

Supported models (config.model.baseline_model):
    linear_reg         - LinearRegression
    random_forest      - RandomForestRegressor
    xgboost            - XGBRegressor  (requires: pip install xgboost)
    gradient_boosting  - GradientBoostingRegressor

Usage:
    from baseline_models import train_baseline_run, test_baseline_run

Author: Aga
"""

import os
import joblib
import numpy as np
from typing import Dict, Tuple, Optional, Any

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from AtomicDataset import AtomicDataset
from test_model import compute_metrics, print_metrics, save_predictions_excel
from utils import (
    set_seed,
    get_experiment_tags,
    append_metrics_to_excel,
    _get_elements_str,
)


# ── Model factory ─────────────────────────────────────────────────────────────

def get_baseline_model(config):
    """
    Return an unfitted sklearn estimator selected by config.model.baseline_model.

    Hyperparameters are set to reasonable defaults; further tuning can be done
    by adding a baseline_params section to config_atomic.yaml.
    """
    model_type = config.model.baseline_model
    seed = config.general.random_seed

    if model_type == "linear_reg":
        return LinearRegression()

    if model_type == "random_forest":
        return RandomForestRegressor(
            n_estimators=200,
            max_features="sqrt",
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=seed,
        )

    if model_type == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            random_state=seed,
        )

    if model_type == "xgboost":
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError(
                "xgboost is not installed. Run: pip install xgboost"
            )
        return xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=seed,
            verbosity=0,
        )

    raise ValueError(
        f"Unknown baseline_model '{model_type}'. "
        "Choose from: linear_reg, random_forest, xgboost, gradient_boosting"
    )


# ── Naming helpers ─────────────────────────────────────────────────────────────

def get_baseline_model_path(config) -> str:
    """
    Generate the checkpoint filename for a baseline model.

    Format: saved_models/baseline_{model_type}_{elements}_{tags}.pkl
    Analogous to get_model_name_from_config() used by the DL pipeline.
    """
    model_type = config.model.baseline_model
    elements_str = _get_elements_str(config)
    tags = get_experiment_tags(config)
    filename = f"baseline_{model_type}_{elements_str}_{tags}.pkl"
    return os.path.join(config.logging.save_dir, filename)


# ── Training ──────────────────────────────────────────────────────────────────

def train_baseline_run(config) -> Tuple[str, Dict[str, float], Dict[str, float]]:
    """
    Train a baseline ML model.

    Mirrors the interface of train_one_run() so run_experiments_baseline.py
    can call it identically to the DL runner.

    Returns:
        (model_path, train_metrics, val_metrics)
    """
    set_seed(config.general.random_seed)
    model_type = config.model.baseline_model

    print(f"\nTraining baseline model: {model_type}")

    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading datasets...")
    train_dataset = AtomicDataset(config, subset="train")
    val_dataset = AtomicDataset(
        config,
        subset="val",
        scaler_features=train_dataset.scaler_features,
        scaler_target=train_dataset.scaler_target,
    )

    X_train, y_train = train_dataset.X, train_dataset.y          # already scaled
    X_val, y_val = val_dataset.X, val_dataset.y

    sample_weights: Optional[np.ndarray] = train_dataset.sample_weights

    print(f"  Train samples : {len(X_train)}, features: {X_train.shape[1]}")
    print(f"  Val   samples : {len(X_val)}")

    # ── Fit model ──────────────────────────────────────────────────────────────
    model = get_baseline_model(config)
    print(f"Fitting {model_type}...")

    fit_kwargs: dict = {}
    if sample_weights is not None:
        # GradientBoosting and RandomForest accept sample_weight in fit();
        # LinearRegression and XGBRegressor also accept it.
        fit_kwargs["sample_weight"] = sample_weights

    model.fit(X_train, y_train.ravel(), **fit_kwargs)
    print(f"  Done.")

    # ── Compute metrics in original scale ──────────────────────────────────────
    train_preds_cm = train_dataset.inverse_transform_target(
        model.predict(X_train).reshape(-1, 1)
    )
    train_targets_cm = train_dataset.inverse_transform_target(y_train)
    train_metrics = compute_metrics(train_preds_cm, train_targets_cm)

    val_preds_cm = val_dataset.inverse_transform_target(
        model.predict(X_val).reshape(-1, 1)
    )
    val_targets_cm = val_dataset.inverse_transform_target(y_val)
    val_metrics = compute_metrics(val_preds_cm, val_targets_cm)

    print(f"  Train MAE: {train_metrics['mae']:.2f} cm⁻¹")
    print(f"  Val   MAE: {val_metrics['mae']:.2f} cm⁻¹  |  R²: {val_metrics['r2']:.4f}")

    # ── Save model ─────────────────────────────────────────────────────────────
    os.makedirs(config.logging.save_dir, exist_ok=True)
    model_path = get_baseline_model_path(config)
    joblib.dump(model, model_path)
    print(f"  Saved model to: {model_path}")

    return model_path, train_metrics, val_metrics


# ── Testing ───────────────────────────────────────────────────────────────────

def test_baseline_run(
    config,
    model_path: str,
    train_metrics: Optional[Dict[str, float]] = None,
    val_metrics: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Evaluate a trained baseline model on the test set.

    Mirrors the interface of test_one_run() — appends metrics and predictions
    to the same Excel files used by the DL pipeline for direct comparison.

    Returns:
        Dictionary of test metrics
    """
    print("\nEvaluating baseline model on test set...")

    # ── Load data (same scalers as training) ────────────────────────────────────
    train_dataset = AtomicDataset(config, subset="train")
    test_dataset = AtomicDataset(
        config,
        subset="test",
        scaler_features=train_dataset.scaler_features,
        scaler_target=train_dataset.scaler_target,
    )

    X_test, y_test = test_dataset.X, test_dataset.y

    # ── Load and run model ─────────────────────────────────────────────────────
    model = joblib.load(model_path)
    preds_raw = model.predict(X_test).reshape(-1, 1)

    # Convert back to original scale (cm⁻¹)
    predictions_cm = test_dataset.inverse_transform_target(preds_raw)
    targets_cm = test_dataset.inverse_transform_target(y_test)

    # ── Metrics ────────────────────────────────────────────────────────────────
    test_metrics = compute_metrics(predictions_cm, targets_cm)
    print_metrics(test_metrics)

    # ── Save predictions column to predictions_{elements}.xlsx ─────────────────
    save_predictions_excel(predictions_cm, targets_cm, test_dataset, config)

    # ── Append metrics row to results_{elements}.xlsx ──────────────────────────
    features_info = {
        "n_features": test_dataset.get_input_dim(),
        "feature_names": test_dataset.get_feature_names(),
    }
    append_metrics_to_excel(
        config,
        test_metrics=test_metrics,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        features=features_info,
    )

    return test_metrics
