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
import pandas as pd
from typing import Tuple, Optional
import time
import os
import json

from AtomicDataset import AtomicDataset
from AtomicModel import create_model
from utils import (
    create_loss_function,
    save_checkpoint,
    format_time, get_model_name_from_config
)


class MultiTaskLoss(nn.Module):
    """
    Combined loss for simultaneous energy and Landé g-factor prediction.

    Physics motivation:
        Atomic energy levels and gJ values are two complementary observables
        of the same quantum state.  Training on both simultaneously forces the
        shared trunk to encode physically meaningful electronic structure features
        that explain BOTH observables, acting as an auxiliary regulariser for the
        primary energy task.

    Gradient-balance design:
        The two losses can be on wildly different scales:
          - unnormalised energy:   L_energy  ≈ 44 000  (SmoothL1, linear region)
          - raw gJ:                L_gJ      ≈ 0.5 – 4  (MSE)

        Naively writing  total = alpha*L_energy + (1-alpha)*L_gJ  does NOT give
        alpha-fraction of gradient from energy, because the gradient magnitudes
        are proportional to the loss values, not to alpha.

        The fix is to pre-scale gJ at criterion-creation time so that both raw
        losses start at approximately the same magnitude:

            gj_loss_multiplier = initial_energy_scale / initial_gJ_scale

        Then the combined loss is:

            total = alpha * L_energy + (1-alpha) * (L_gJ × multiplier)

        After this, alpha truly controls the gradient fraction:
            ∂total/∂θ_energy  ∝  alpha × ∂L_energy/∂θ
            ∂total/∂θ_gJ      ∝  (1-alpha) × multiplier × ∂L_gJ/∂θ

        and both terms are of comparable magnitude.

        Note: dividing losses by a running scale (as in some multi-task papers)
        does NOT work when SmoothL1 is in its linear region, because the gradient
        of L_energy w.r.t. θ_energy is ∝ 1 (constant), so dividing L_energy by
        its running scale s_e gives ∂(L_e/s_e)/∂θ_e ∝ 1/s_e — too small when
        s_e is large.  The multiplier approach avoids this by never dividing.

    Args:
        alpha:                 Weight for energy task (0.0 – 1.0).  Default 0.9.
        energy_criterion:      Loss type for energy head ('MSE', 'MAE', 'Huber', 'SmoothL1').
        gj_criterion:          Loss type for gJ head   ('MSE', 'MAE', 'Huber').
        ema_decay:             EMA decay for monitoring buffers (not used in loss computation).
        initial_energy_scale:  Initial value for the energy EMA buffer (monitoring only).
        initial_gj_scale:      Initial value for the gJ EMA buffer (monitoring only).
        gj_loss_multiplier:    Pre-scale factor applied to gJ loss before combining.
                               Computed as initial_energy_scale / initial_gJ_scale at
                               training start so both terms start at similar magnitude.
    """

    def __init__(self, alpha: float = 0.9,
                 energy_criterion: str = 'MSE',
                 gj_criterion: str = 'MSE',
                 ema_decay: float = 0.95,
                 initial_energy_scale: float = 1.0,
                 initial_gj_scale: float = 1.0,
                 gj_loss_multiplier: float = 1.0):
        super(MultiTaskLoss, self).__init__()
        self.alpha = alpha
        self.ema_decay = ema_decay
        # gj_loss_multiplier: fixed at training start, brings gJ loss to energy magnitude.
        # After scaling: (1-alpha)*L_gJ_scaled and alpha*L_energy are comparable, so
        # alpha correctly controls the gradient fraction for each task.
        self.gj_loss_multiplier = gj_loss_multiplier

        # EMA buffers for monitoring — track scaled losses so we can print
        # gJ_scale_ratio = gj_loss_scale / energy_loss_scale each epoch.
        # They do NOT enter the backward pass; the multiplier is the only balancing term.
        self.register_buffer('energy_loss_scale', torch.tensor(float(initial_energy_scale)))
        self.register_buffer('gj_loss_scale',     torch.tensor(float(initial_gj_scale * gj_loss_multiplier)))

        # Energy loss: mean over batch (no masking — all rows have energy labels)
        self.energy_criterion = create_loss_function(energy_criterion, reduction='mean')

        # gJ loss: element-wise so we can zero out unobserved rows before averaging
        self.gj_criterion_none = create_loss_function(gj_criterion, reduction='none')

    def forward(
        self,
        energy_pred:   torch.Tensor,   # (B, 1)  normalised energy prediction
        energy_target: torch.Tensor,   # (B, 1)  normalised energy target
        gj_pred:       torch.Tensor,   # (B, 1)  raw gJ prediction
        gj_target:     torch.Tensor,   # (B, 1)  raw experimental gJ (0.0 where unobserved)
        gj_mask:       torch.Tensor,   # (B, 1)  float32: 1.0=observed, 0.0=missing
    ):
        """
        Compute the combined multi-task loss using pre-scaled gJ term.

        The gj_loss_multiplier (= initial_energy_scale / initial_gj_scale) was fixed
        at criterion-creation time to bring the gJ loss to the same order of magnitude
        as the energy loss.  This makes alpha behave as advertised — the fraction of
        gradient from the energy task — regardless of whether target normalisation is
        enabled and regardless of the raw loss scales.

        Crucially, neither loss is divided by any running scale.  Raw gradients flow
        through backward() at their natural magnitude; only the mixing ratio changes.

        Returns:
            total_loss  — scalar tensor, used for backward()
            energy_loss — Python float (raw, unscaled), for logging in physical units
            gj_loss     — Python float (raw, unscaled), for logging in physical units
        """
        # --- Energy loss: mean over all batch rows (no masking needed) ---
        energy_loss = self.energy_criterion(energy_pred, energy_target)

        # --- gJ loss: masked mean over rows where experimental gJ is available ---
        gj_elementwise = self.gj_criterion_none(gj_pred, gj_target)   # shape (B, 1)
        mask_sum = gj_mask.sum()
        if mask_sum > 0:
            # Only rows with observed gJ contribute — avoids bias from missing labels
            gj_loss = (gj_elementwise * gj_mask).sum() / mask_sum
        else:
            # Entire batch has no observed gJ; gJ term is silenced this step
            gj_loss = torch.tensor(0.0, device=energy_pred.device)

        # --- Pre-scale gJ loss to match energy magnitude ---
        # multiplier is a fixed scalar set at training start — does not change per step.
        # Scaling happens BEFORE combining with energy_loss, so both terms contribute
        # gradients of comparable magnitude to the trunk.
        gj_loss_scaled = gj_loss * self.gj_loss_multiplier

        # --- EMA monitoring (detached — does NOT affect backward pass) ---
        # Tracks running average of each scaled term so we can print gJ_scale_ratio.
        # gJ_scale_ratio = gj_loss_scale / energy_loss_scale should stay near 1.0
        # throughout training if the multiplier is calibrated correctly.
        if self.training:
            with torch.no_grad():
                self.energy_loss_scale.mul_(self.ema_decay).add_(
                    (1.0 - self.ema_decay) * energy_loss.detach())
                if mask_sum > 0:
                    self.gj_loss_scale.mul_(self.ema_decay).add_(
                        (1.0 - self.ema_decay) * gj_loss_scaled.detach())

        # --- Combine with alpha weighting ---
        # After pre-scaling, energy_loss ≈ gj_loss_scaled ≈ O(1), so alpha is meaningful:
        #   alpha=0.9 → 90% of gradient from energy, 10% from gJ (approximately)
        total_loss = self.alpha * energy_loss + (1.0 - self.alpha) * gj_loss_scaled

        # Return raw (unscaled) gJ loss for logging in physical units
        return total_loss, energy_loss.item(), gj_loss.item()


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

    # Detect multi-task mode once per epoch (avoids repeated config lookups)
    is_multitask = config.training.get('multitask_gj', False)

    running_loss = 0.0
    all_predictions = []    # energy predictions (used for epoch MAE)
    all_targets = []        # energy targets

    # Multi-task: also accumulate gJ losses for per-epoch reporting
    epoch_gj_losses = []    # individual batch gJ loss values (floats)

    use_focal = config.training.get('use_focal_loss', False)
    focal_alpha = config.training.get('focal_loss_alpha', 0.5)
    use_sample_weights = (
            hasattr(train_loader.dataset, 'sample_weights')
            and train_loader.dataset.sample_weights is not None
    )

    # Iterate through batches of training data
    for batch_idx, batch in enumerate(train_loader):

        # ----------------------------------------------------------------
        # Unpack batch — shape differs between single-task and multi-task
        # ----------------------------------------------------------------
        if is_multitask:
            # DataLoader returns 4-tuple: (features, y_energy, y_gj, gj_mask)
            features, targets_energy, targets_gj, gj_mask = batch
            features       = features.to(device)
            targets_energy = targets_energy.to(device)
            targets_gj     = targets_gj.to(device)
            gj_mask        = gj_mask.to(device)
        else:
            # Single-task: standard 2-tuple (features, targets)
            features, targets = batch
            features = features.to(device)
            targets  = targets.to(device)

        # Zero out gradients from previous iteration
        # PyTorch accumulates gradients, so we must clear them each time
        optimizer.zero_grad()

        # ----------------------------------------------------------------
        # Forward pass + loss computation
        # ----------------------------------------------------------------
        if is_multitask:
            # Model returns (energy_pred, gj_pred) for MultiTaskAtomicModel
            pred_energy, pred_gj = model(features)

            # MultiTaskLoss combines energy and masked gJ losses with weighting α
            total_loss, e_loss_val, g_loss_val = criterion(
                pred_energy, targets_energy, pred_gj, targets_gj, gj_mask
            )
            loss = total_loss   # scalar tensor — drives backward()
            epoch_gj_losses.append(g_loss_val)

            # Track energy predictions for epoch-level MAE (single-task compatible)
            all_predictions.append(pred_energy.detach().cpu().numpy())
            all_targets.append(targets_energy.cpu().numpy())

        else:
            # Single-task path: IDENTICAL to original code
            # Forward pass: compute predictions
            predictions = model(features)

            # Compute per-sample losses — shape: [batch_size, 1]
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

            all_predictions.append(predictions.detach().cpu().numpy())
            all_targets.append(targets.cpu().numpy())

        # ----------------------------------------------------------------
        # Backward pass and optimisation (identical for both modes)
        # ----------------------------------------------------------------
        # Backward pass: compute gradients — how much each weight contributed to loss
        loss.backward()

        # Gradient clipping: prevent exploding gradients
        # If gradients become too large, they can destabilize training
        if config.training.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.training.gradient_clip
            )

        # Optimization step: update weights based on gradients
        optimizer.step()

        # Track total loss for this batch (same meaning in both modes)
        running_loss += loss.item()

    # Average loss across all batches
    avg_loss = running_loss / len(train_loader)

    # Combine all energy predictions for epoch-level MAE (both modes)
    all_predictions = np.concatenate(all_predictions)
    all_targets     = np.concatenate(all_targets)

    # Inverse transform BEFORE computing MAE (undo StandardScaler + any target transform)
    predictions_cm = train_loader.dataset.inverse_transform_target(all_predictions)
    targets_cm     = train_loader.dataset.inverse_transform_target(all_targets)

    # Calculate average energy MAE in cm⁻¹
    avg_mae = np.mean(np.abs(predictions_cm - targets_cm))
    # RuntimeWarning: invalid value encountered in subtract

    # Average gJ loss across batches (nan in single-task mode — not used there)
    avg_gj_loss = float(np.mean(epoch_gj_losses)) if epoch_gj_losses else float('nan')

    return avg_loss, avg_mae, avg_gj_loss


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    val_dataset: AtomicDataset,
    config=None,
) -> Tuple[float, float, float, float]:
    """
    Evaluate the model on validation data.

    Validation checks how well the model generalizes to unseen data.
    This is done without updating weights (no training).

    Args:
        model:      Neural network model
        val_loader: DataLoader for validation data
        criterion:  Loss function (MultiTaskLoss in multitask mode)
        device:     CPU or GPU
        val_dataset: Dataset object used for inverse transform
        config:     Configuration object (needed for multitask detection)

    Returns:
        Tuple of (loss, mae_energy, rmse_energy, mae_gj)
        - loss:        Average validation loss (total, same for both modes)
        - mae_energy:  Mean Absolute Error for energy (cm⁻¹)
        - rmse_energy: Root Mean Squared Error for energy (cm⁻¹)
        - mae_gj:      Mean Absolute Error for gJ (dimensionless);
                       nan if single-task mode or no observed gJ in val set
    """
    # Set model to evaluation mode (disables dropout; batch norm uses running stats)
    model.eval()

    is_multitask = (config is not None) and config.training.get('multitask_gj', False)

    running_loss = 0.0
    all_predictions = []   # energy predictions
    all_targets     = []   # energy targets
    num_batches = 0

    # Multi-task: collect gJ predictions to compute gJ MAE over observed rows
    all_gj_preds   = []
    all_gj_targets = []
    all_gj_masks   = []

    # Disable gradient computation — only evaluating, not training
    with torch.no_grad():
        for batch in val_loader:

            if is_multitask:
                features, targets_energy, targets_gj, gj_mask = batch
                features       = features.to(device)
                targets_energy = targets_energy.to(device)
                targets_gj     = targets_gj.to(device)
                gj_mask        = gj_mask.to(device)

                pred_energy, pred_gj = model(features)

                # MultiTaskLoss returns (total_loss_tensor, e_loss_float, gj_loss_float)
                total_loss, _, _ = criterion(
                    pred_energy, targets_energy, pred_gj, targets_gj, gj_mask
                )
                running_loss += total_loss.item()

                all_predictions.append(pred_energy.cpu().numpy())
                all_targets.append(targets_energy.cpu().numpy())
                all_gj_preds.append(pred_gj.cpu().numpy())
                all_gj_targets.append(targets_gj.cpu().numpy())
                all_gj_masks.append(gj_mask.cpu().numpy())

            else:
                # Single-task path — identical to original validate()
                features, targets = batch
                features = features.to(device)
                targets  = targets.to(device)

                predictions = model(features)
                loss = criterion(predictions, targets)
                running_loss += loss.item()

                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

            num_batches += 1

    # Combine all batches
    all_predictions = np.concatenate(all_predictions)
    all_targets     = np.concatenate(all_targets)

    avg_loss = running_loss / num_batches

    # Inverse transform BEFORE computing energy metrics
    predictions_cm = val_dataset.inverse_transform_target(all_predictions)
    targets_cm     = val_dataset.inverse_transform_target(all_targets)

    # RuntimeWarning: invalid value encountered in subtract
    mae  = np.mean(np.abs(predictions_cm - targets_cm))
    rmse = np.sqrt(np.mean((predictions_cm - targets_cm) ** 2))
    # RuntimeWarning: overflow encountered in square

    # gJ MAE — only meaningful in multitask mode with at least one observed row
    if is_multitask and all_gj_preds:
        all_gj_preds   = np.concatenate(all_gj_preds).flatten()   # (N,)
        all_gj_targets = np.concatenate(all_gj_targets).flatten()  # (N,)
        all_gj_masks   = np.concatenate(all_gj_masks).flatten()    # (N,)
        observed = all_gj_masks == 1.0
        if observed.sum() > 0:
            mae_gj = float(np.mean(np.abs(all_gj_preds[observed] - all_gj_targets[observed])))
        else:
            mae_gj = float('nan')  # no observed gJ in validation set
    else:
        mae_gj = float('nan')  # single-task mode — gJ MAE not applicable

    return avg_loss, mae, rmse, mae_gj


def debug_multitask_setup(model, train_loader, criterion_train, optimizer, device, config):
    """
    Run a single forward+backward pass and print a full diagnostic report.

    Checks the following failure modes (in order of likelihood given the
    symptoms: energy MAE frozen at dataset mean, gJ MAE learning):

      [1] Output shapes  — confirms model returns (energy, gJ) tuple not a scalar
      [2] Prediction ranges  — confirms heads are not saturated at init
      [3] Buffer type check  — confirms EMA scales are still torch.Tensor
                               (plain assignment breaks register_buffer)
      [4] Raw loss values  — shows actual energy_loss and gJ_loss magnitudes
      [5] EMA update check  — confirms scales moved after one forward pass
      [6] Normalised contributions  — shows actual alpha split reaching optimiser
      [7] Per-parameter gradients  — shows which layers receive non-zero gradients
                                     and whether energy/gJ heads are connected
      [8] Single-step sanity  — confirms loss decreases with one SGD step

    Args:
        model:            the MultiTaskAtomicModel (will be re-created before return)
        train_loader:     DataLoader for training set
        criterion_train:  MultiTaskLoss instance
        optimizer:        the real optimizer (used only to read config; not stepped)
        device:           torch.device
        config:           the full config object

    Returns:
        A freshly re-initialised model with the same architecture.
        Use this model for the actual training loop.
    """

    SEP = "=" * 65

    print(f"\n{SEP}")
    print("  MULTITASK TRAINING DIAGNOSTICS")
    print(SEP)

    model.train()
    criterion_train.train()

    # ------------------------------------------------------------------ #
    # Grab one real batch                                                  #
    # ------------------------------------------------------------------ #
    batch = next(iter(train_loader))
    features, targets_e, targets_gj, gj_mask = [b.to(device) for b in batch]

    # ------------------------------------------------------------------ #
    # [1] Output shapes                                                    #
    # ------------------------------------------------------------------ #
    print("\n[1] Model output shapes")
    with torch.no_grad():
        out = model(features)

    if not isinstance(out, tuple) or len(out) != 2:
        print(f"  ✗ CRITICAL: model returned {type(out)}, expected (energy, gj) tuple")
        print(f"    Check MultiTaskAtomicModel.forward() — it must return (energy_pred, gj_pred)")
        print(f"    Aborting diagnostics.\n{SEP}\n")
        return _fresh_model(model, config, features.shape[1], device)

    pred_e, pred_gj = out
    print(f"  pred_e  shape : {tuple(pred_e.shape)}  ← energy head")
    print(f"  pred_gj shape : {tuple(pred_gj.shape)}  ← gJ head")
    ok_shapes = (pred_e.shape == targets_e.shape and pred_gj.shape == targets_gj.shape)
    print(f"  Shapes match targets: {ok_shapes}  {'✓' if ok_shapes else '✗ shape mismatch — check head output dims'}")

    # ------------------------------------------------------------------ #
    # [2] Prediction ranges at init                                        #
    # ------------------------------------------------------------------ #
    print("\n[2] Prediction ranges at initialisation")
    print(f"  pred_e   min={pred_e.min().item():10.2f}  max={pred_e.max().item():10.2f}  "
          f"mean={pred_e.mean().item():10.2f}")
    print(f"  target_e min={targets_e.min().item():10.2f}  max={targets_e.max().item():10.2f}  "
          f"mean={targets_e.mean().item():10.2f}")
    print(f"  pred_gj  min={pred_gj.min().item():8.4f}  max={pred_gj.max().item():8.4f}  "
          f"mean={pred_gj.mean().item():8.4f}")
    n_obs = int(gj_mask.sum().item())
    if n_obs > 0:
        obs_tgt = targets_gj[gj_mask.squeeze() == 1]
        print(f"  target_gj (observed only, N={n_obs})  "
              f"min={obs_tgt.min().item():.4f}  max={obs_tgt.max().item():.4f}")
    else:
        print(f"  ⚠ No observed gJ rows in this batch (mask all zeros)")

    # ------------------------------------------------------------------ #
    # [3] Buffer type check — catches plain-assignment EMA bug             #
    # ------------------------------------------------------------------ #
    print("\n[3] EMA scale buffer type check")
    e_scale_attr = getattr(criterion_train, 'energy_loss_scale', None)
    g_scale_attr = getattr(criterion_train, 'gj_loss_scale', None)

    e_is_tensor = isinstance(e_scale_attr, torch.Tensor)
    g_is_tensor = isinstance(g_scale_attr, torch.Tensor)

    print(f"  energy_loss_scale : value={float(e_scale_attr):.4f}  "
          f"type={type(e_scale_attr).__name__}  "
          f"{'✓ Tensor' if e_is_tensor else '✗ NOT a Tensor — plain assignment broke register_buffer!'}")
    print(f"  gj_loss_scale     : value={float(g_scale_attr):.4f}  "
          f"type={type(g_scale_attr).__name__}  "
          f"{'✓ Tensor' if g_is_tensor else '✗ NOT a Tensor — plain assignment broke register_buffer!'}")

    if not e_is_tensor or not g_is_tensor:
        print("\n  FIX: in MultiTaskLoss.forward(), replace")
        print("    self.energy_loss_scale = self.ema_decay * self.energy_loss_scale + ...")
        print("  with")
        print("    self.energy_loss_scale.mul_(self.ema_decay).add_((1-self.ema_decay)*energy_loss.detach())")

    # ------------------------------------------------------------------ #
    # [4] Raw loss values and scaled gJ loss                              #
    # ------------------------------------------------------------------ #
    print("\n[4] Raw loss values and multiplier scaling")
    # Need gradients for backward check in [7]
    optimizer.zero_grad()
    pred_e2, pred_gj2 = model(features)
    total_loss, e_loss_val, gj_loss_val = criterion_train(
        pred_e2, targets_e, pred_gj2, targets_gj, gj_mask
    )
    mult = getattr(criterion_train, 'gj_loss_multiplier', 1.0)
    gj_loss_scaled_val = gj_loss_val * mult
    ratio_to_energy = gj_loss_scaled_val / (e_loss_val + 1e-8)
    print(f"  energy_loss (raw)    : {e_loss_val:.4f}")
    print(f"  gj_loss (raw)        : {gj_loss_val:.6f}")
    print(f"  gj_loss (scaled)     : {gj_loss_scaled_val:.4f}  (×multiplier={mult:.2f})")
    print(f"  ratio scaled_gJ/energy (should be ≈1): {ratio_to_energy:.4f}  "
          f"{'✓' if 0.1 <= ratio_to_energy <= 10 else '⚠ far from 1.0 — multiplier may be wrong'}")
    print(f"  total_loss           : {total_loss.item():.6f}")

    # ------------------------------------------------------------------ #
    # [5] EMA update check                                                 #
    # ------------------------------------------------------------------ #
    print("\n[5] EMA scale update check")
    e_scale_after = float(getattr(criterion_train, 'energy_loss_scale', 0))
    g_scale_after = float(getattr(criterion_train, 'gj_loss_scale', 0))
    print(f"  energy_loss_scale after forward: {e_scale_after:.4f}  (raw loss was {e_loss_val:.4f})")
    print(f"  gj_loss_scale     after forward: {g_scale_after:.6f}  (raw loss was {gj_loss_val:.6f})")

    # If plain assignment bug: scale == initial value (unchanged), or is a float
    e_updated = e_is_tensor and abs(e_scale_after - e_loss_val) < abs(e_scale_after - 44363)
    g_updated = g_is_tensor and (n_obs == 0 or abs(g_scale_after - gj_loss_val) < abs(g_scale_after - 3.638))
    print(f"  EMA tracking energy: {e_updated}  {'✓' if e_updated else '⚠ scale unchanged — EMA not updating'}")
    print(f"  EMA tracking gJ    : {g_updated}  {'✓' if g_updated else '⚠ scale unchanged or no observed gJ'}")

    # ------------------------------------------------------------------ #
    # [6] Effective gradient contribution split                           #
    # ------------------------------------------------------------------ #
    print("\n[6] Effective gradient contribution split")
    alpha = criterion_train.alpha
    mult  = getattr(criterion_train, 'gj_loss_multiplier', 1.0)
    # With the multiplier formulation, contributions are:
    #   energy: alpha × L_energy
    #   gJ:     (1-alpha) × L_gJ × multiplier
    e_contrib  = alpha * e_loss_val
    g_contrib  = (1.0 - alpha) * gj_loss_val * mult
    total_contrib = e_contrib + g_contrib + 1e-12
    print(f"  alpha={alpha:.3f}   multiplier={mult:.2f}")
    print(f"  energy contrib  = {alpha:.3f} × {e_loss_val:.4f}            = {e_contrib:.4f}  "
          f"({100*e_contrib/total_contrib:.1f}% of total)")
    print(f"  gJ contrib      = {1-alpha:.3f} × {gj_loss_val:.6f} × {mult:.2f} = {g_contrib:.4f}  "
          f"({100*g_contrib/total_contrib:.1f}% of total)")
    if 100*g_contrib/total_contrib < 1.0:
        print(f"  ⚠ gJ share < 1% — gJ head is effectively receiving no gradient")
        print(f"    Check: is gj_loss_multiplier correct?  multiplier={mult:.4f}")
    elif 100*e_contrib/total_contrib < 10.0:
        print(f"  ⚠ energy share < 10% — energy head may be starved")
        print(f"    Consider: higher alpha or smaller gj_loss_multiplier")
    else:
        print(f"  ✓ Both heads receive meaningful gradient signal")

    # ------------------------------------------------------------------ #
    # [7] Per-parameter gradient magnitudes                                #
    # ------------------------------------------------------------------ #
    print("\n[7] Per-parameter gradient magnitudes")
    total_loss.backward()

    energy_head_grads = []
    gj_head_grads     = []
    trunk_grads       = []
    no_grad_params    = []

    for name, param in model.named_parameters():
        if param.grad is None:
            no_grad_params.append(name)
            continue
        gnorm = param.grad.abs().mean().item()   # mean abs gradient (more stable than norm for comparison)
        if 'energy_head' in name:
            energy_head_grads.append((name, gnorm))
        elif 'gj_head' in name:
            gj_head_grads.append((name, gnorm))
        else:
            trunk_grads.append((name, gnorm))

    if no_grad_params:
        print(f"  ✗ Parameters with NO gradient (disconnected):")
        for n in no_grad_params:
            print(f"      {n}")

    print(f"  Trunk (shared) layers:")
    for name, g in trunk_grads:
        flag = "⚠ near zero" if g < 1e-7 else ""
        print(f"    {name:48s}  |grad|_mean={g:.2e}  {flag}")

    print(f"  Energy head:")
    if energy_head_grads:
        for name, g in energy_head_grads:
            flag = "⚠ near zero" if g < 1e-7 else ""
            print(f"    {name:48s}  |grad|_mean={g:.2e}  {flag}")
    else:
        print(f"    ✗ NO energy_head parameters found in model")
        print(f"      Model parameter names: {[n for n,_ in model.named_parameters()]}")
        print(f"      Check that energy head is named 'energy_head.*' in MultiTaskAtomicModel")

    print(f"  gJ head:")
    if gj_head_grads:
        for name, g in gj_head_grads:
            flag = "⚠ near zero" if g < 1e-7 else ""
            print(f"    {name:48s}  |grad|_mean={g:.2e}  {flag}")
    else:
        print(f"    ✗ NO gj_head parameters found in model")
        print(f"      Check that gJ head is named 'gj_head.*' in MultiTaskAtomicModel")

    if energy_head_grads and gj_head_grads:
        e_total = sum(g for _, g in energy_head_grads)
        g_total = sum(g for _, g in gj_head_grads)
        ratio   = e_total / (g_total + 1e-12)
        target_ratio = alpha / (1 - alpha + 1e-12)
        print(f"\n  Energy/gJ head gradient ratio : {ratio:.2f}  "
              f"(target after normalisation ≈ alpha/(1-alpha) = {target_ratio:.1f})")
        if ratio > target_ratio * 100:
            print(f"  ⚠ Energy still dominates by {ratio/target_ratio:.0f}× "
                  f"— normalisation is not working correctly")

    # ------------------------------------------------------------------ #
    # [8] Single-step sanity check                                         #
    # ------------------------------------------------------------------ #
    print("\n[8] Single-step sanity check")
    print(f"  Before: energy_loss={e_loss_val:.2f}  gj_loss={gj_loss_val:.6f}  total={total_loss.item():.6f}")

    # SGD step with a modest LR — should decrease both losses
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                param.data.add_(param.grad, alpha=-1e-5)   # manual SGD step, lr=1e-5

    with torch.no_grad():
        pred_e3, pred_gj3 = model(features)

    # Use a fresh criterion instance (so EMA is at init values, for a fair before/after comparison).
    # Pass the same gj_loss_multiplier so the total loss is computed identically.
    crit_check = criterion_train.__class__(
        alpha=alpha,
        energy_criterion=config.training.criterion,
        gj_criterion=config.training.get('gj_criterion', 'MSE'),
        initial_energy_scale=e_loss_val,
        initial_gj_scale=max(gj_loss_val, 1e-6),
        gj_loss_multiplier=getattr(criterion_train, 'gj_loss_multiplier', 1.0),
    )
    crit_check.eval()
    with torch.no_grad():
        total3, e3, gj3 = crit_check(pred_e3, targets_e, pred_gj3, targets_gj, gj_mask)

    print(f"  After : energy_loss={e3:.4f}  gj_loss={gj3:.6f}  total={total3.item():.6f}")
    e_dec   = e3 < e_loss_val
    gj_dec  = gj3 < gj_loss_val or n_obs == 0
    tot_dec = total3.item() < total_loss.item()
    print(f"  Energy loss decreased : {e_dec}   {'✓' if e_dec else '✗ energy head not learning'}")
    print(f"  gJ loss decreased     : {gj_dec}  {'✓' if gj_dec else '✗ gJ head not learning'}")
    print(f"  Total loss decreased  : {tot_dec} {'✓' if tot_dec else '✗ backward pass broken'}")

    print(f"\n{SEP}")
    print("  END DIAGNOSTICS — returning fresh model for clean training start")
    print(SEP + "\n")

    # Re-create a fresh model (diagnostic stepped the weights — don't train from there)
    fresh = _fresh_model(model, config, features.shape[1], device)
    return fresh


def _fresh_model(original_model, config, input_dim, device):
    """Re-create a fresh model with the same architecture, re-initialised weights."""
    fresh = create_model(config, input_dim).to(device)
    print(f"  Fresh model created (input_dim={input_dim}, "
          f"params={sum(p.numel() for p in fresh.parameters()):,})")
    return fresh


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
    # Create loss function(s).
    # In multi-task mode both train and val use MultiTaskLoss, which handles
    # masking internally.  In single-task mode the existing per-element loss
    # (reduction='none') and mean loss (reduction='mean') are used unchanged.
    is_multitask = config.training.get('multitask_gj', False)
    if is_multitask:
        alpha   = config.training.get('multitask_alpha', 0.9)
        gj_crit = config.training.get('gj_criterion', 'MSE')

        # ----------------------------------------------------------------
        # Estimate initial loss scales with a forward pass on ≤10 batches.
        #
        # Why: the gj_loss_multiplier = initial_energy_scale / initial_gJ_scale
        # pre-scales the gJ term so that both raw losses start at comparable
        # magnitude.  This makes alpha correctly control the gradient fraction
        # for each task regardless of whether target normalisation is on/off.
        # ----------------------------------------------------------------
        print(f"\nMulti-task mode: predicting energy (α={alpha:.2f}) + gJ (α={1 - alpha:.2f})")
        print(f"  Energy criterion: {config.training.criterion} | gJ criterion: {gj_crit}")
        print(f"  Estimating initial loss scales (10 training batches)...")

        model.eval()   # eval mode so BN/dropout don't interfere with the scale estimate
        with torch.no_grad():
            sample_e_losses  = []
            sample_gj_losses = []
            # Use criterion functions with the exact same reduction settings that
            # MultiTaskLoss will use internally, so the numbers are comparable.
            temp_e_crit  = create_loss_function(config.training.criterion, reduction='mean')
            temp_gj_crit = create_loss_function(gj_crit, reduction='none')

            for i, batch in enumerate(train_loader):
                features, targets_e, targets_gj, gj_mask = [b.to(device) for b in batch]
                pred_e, pred_gj = model(features)

                sample_e_losses.append(temp_e_crit(pred_e, targets_e).item())

                mask_sum = gj_mask.sum()
                if mask_sum > 0:
                    # Masked mean — same as MultiTaskLoss.forward computes
                    gj_el = temp_gj_crit(pred_gj, targets_gj)   # (B, 1)
                    sample_gj_losses.append(((gj_el * gj_mask).sum() / mask_sum).item())

                if i >= 9:
                    break

        model.train()  # restore training mode before creating the real criteria

        initial_energy_scale = float(np.mean(sample_e_losses)) if sample_e_losses else 1.0
        initial_gj_scale     = float(np.mean(sample_gj_losses)) if sample_gj_losses else 1.0

        # The multiplier brings gJ loss to the same order of magnitude as energy loss.
        # Dividing by scale is intentionally avoided here: raw gradients flow through
        # backward() at their natural magnitude; only the mixing ratio is adjusted.
        gj_loss_multiplier = initial_energy_scale / (initial_gj_scale + 1e-8)

        print(f"  Initial scales — energy: {initial_energy_scale:.4f}, "
              f"gJ: {initial_gj_scale:.4f}, "
              f"gJ multiplier: {gj_loss_multiplier:.4f}")
        print(f"  Scaled gJ loss at init ≈ {initial_gj_scale * gj_loss_multiplier:.4f}  "
              f"(target: ≈ energy loss = {initial_energy_scale:.4f})")

        # ---- Task 5 sanity assertion ----
        # Verify the multiplier actually brought the losses to comparable scale.
        # If this fires, initial_gj_scale was 0 or NaN — check the lande_g column.
        assert 0.1 <= (initial_gj_scale * gj_loss_multiplier) / (initial_energy_scale + 1e-8) <= 10.0, (
            f"gJ multiplier ({gj_loss_multiplier:.2f}) does not bring losses to comparable scale. "
            f"energy_scale={initial_energy_scale:.4f}, "
            f"gJ_scale×multiplier={initial_gj_scale * gj_loss_multiplier:.4f}  "
            f"(ratio must be in [0.1, 10])"
        )
        print(f"  ✓ Loss scales balanced: energy={initial_energy_scale:.4f}, "
              f"gJ×mult={initial_gj_scale * gj_loss_multiplier:.4f}")

        def _make_multitask_criterion():
            """Helper to create a fresh MultiTaskLoss with the calibrated multiplier."""
            return MultiTaskLoss(
                alpha=alpha,
                energy_criterion=config.training.criterion,
                gj_criterion=gj_crit,
                ema_decay=0.95,
                initial_energy_scale=initial_energy_scale,
                initial_gj_scale=initial_gj_scale,
                gj_loss_multiplier=gj_loss_multiplier,
            )

        criterion_train = _make_multitask_criterion()
        criterion_val   = _make_multitask_criterion()  # separate instance, identical config
    else:
        criterion_train = create_loss_function(config.training.criterion, reduction='none')
        criterion_val   = create_loss_function(config.training.criterion, reduction='mean')
        gj_loss_multiplier = 1.0   # unused in single-task, defined for scope safety
    
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

    if is_multitask and config.training.get('debug_multitask', False):
        model = debug_multitask_setup(
            model, train_loader, criterion_train, optimizer, device, config
        )
        # Re-estimate scales on the fresh model (debug contaminated both model and criterion)
        with torch.no_grad():
            sample_e, sample_gj = [], []
            temp_e = create_loss_function(config.training.criterion, reduction='mean')
            temp_gj = create_loss_function(gj_crit, reduction='none')
            for i, batch in enumerate(train_loader):
                features, targets_e, targets_gj, gj_mask = [b.to(device) for b in batch]
                pred_e, pred_gj = model(features)
                sample_e.append(temp_e(pred_e, targets_e).item())
                mask_sum = gj_mask.sum()
                if mask_sum > 0:
                    gj_el = temp_gj(pred_gj, targets_gj)
                    sample_gj.append(((gj_el * gj_mask).sum() / mask_sum).item())
                if i >= 9: break
        initial_energy_scale = float(np.mean(sample_e))
        initial_gj_scale = float(np.mean(sample_gj)) if sample_gj else 3.6

        gj_loss_multiplier = initial_energy_scale / (initial_gj_scale + 1e-8)
        criterion_train = _make_multitask_criterion()
        criterion_val   = _make_multitask_criterion()
        print(f"  Re-estimated after debug — energy: {initial_energy_scale:.4f}, "
              f"gJ: {initial_gj_scale:.4f}, multiplier: {gj_loss_multiplier:.4f}")
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
    best_train_metrics = {}
    best_val_metrics = {}
    
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
        
        # Train for one epoch — returns (avg_loss, energy_mae, gj_loss_avg)
        train_loss, train_mae, train_gj_loss = train_one_epoch(
            model, train_loader, criterion_train, optimizer, device, epoch, config
        )

        # Validate — returns (loss, mae_energy, rmse_energy, mae_gj)
        val_loss, val_mae, val_rmse, val_mae_gj = validate(
            model, val_loader, criterion_val, device, val_dataset, config
        )
        
        # Update learning rate if using scheduler
        if scheduler is not None:
            scheduler.step(val_loss)
        
        epoch_time = time.time() - epoch_start
        
        # Print progress
        if epoch % config.logging.log_interval == 0 or epoch == 1:
            log_line = (
                f"Epoch {epoch:3d}/{config.general.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Train MAE: {train_mae:.4f} cm⁻¹ | "
                f"Val MAE: {val_mae:.2f} cm⁻¹ | "
                f"Val RMSE: {val_rmse:.2f} cm⁻¹ | "
            )
            # In multi-task mode append gJ MAE and scale-ratio health indicator
            if config.training.get('multitask_gj', False):
                if not np.isnan(val_mae_gj):
                    log_line += f"Val gJ MAE: {val_mae_gj:.4f} | "
                # gJ_scale_ratio = gj_loss_scale / energy_loss_scale
                # Both EMA buffers track the scaled (multiplied) gJ loss and energy loss,
                # so the ratio should stay near 1.0 throughout training when the
                # multiplier is calibrated correctly.  Values far from [0.5, 2.0]
                # signal that one task is drifting and may need alpha re-tuning.
                e_ema = float(criterion_train.energy_loss_scale)
                g_ema = float(criterion_train.gj_loss_scale)
                scale_ratio = g_ema / (e_ema + 1e-8)
                log_line += f"gJ_scale_ratio: {scale_ratio:.3f} | "
            log_line += f"Time: {format_time(epoch_time)}"
            print(log_line)
        
        # Check if this is the best model so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_train_metrics = {'loss': train_loss, 'mae': train_mae}
            best_val_metrics = {
                'loss': val_loss, 'mae': val_mae, 'rmse': val_rmse,
                'mae_gj': val_mae_gj,  # nan in single-task mode
            }

            # Save the best model
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss, best_model_path
            )
            # print(f"  → New best model! Val Loss: {val_loss:.4f}")
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
    
    return best_model_path, best_train_metrics, best_val_metrics


# =============================================================================
# MULTI-TASK TESTING CHECKLIST
# Run these manual checks after any change to the multi-task code path.
# =============================================================================
#
# [ ] Single-task run (multitask_gj: false) produces identical results to before:
#     - train_one_epoch returns 3 values; third value (avg_gj_loss) is nan
#     - validate returns 4 values; fourth value (mae_gj) is nan
#     - epoch log line does NOT include "Val gJ MAE"
#     - model file has NO '_mtask' suffix
#
# [ ] Multi-task run starts without error with Co data (266/306 rows have lande_g):
#     - MultiTaskLoss prints "Using loss function: ..." twice (energy + gJ)
#     - "Multi-task mode: predicting energy (α=0.90) + gJ (α=0.10)" line printed
#     - "_prepare_gj_target" prints gJ observed counts for train, val, test
#
# [ ] Val gJ MAE is printed each log_interval epoch in multi-task mode
#
# [ ] gJ predictions are saved in the output CSV (columns: gj_predicted,
#     gj_observed, gj_mask) via test_model.py / save_predictions_excel
#
# [ ] Model file has '_mtask' suffix in multitask mode (e.g., best_model_Co_..._mtask.pt)
#
# [ ] alpha=1.0 in multitask mode reproduces single-task energy loss exactly:
#     MultiTaskLoss.forward → total = 1.0 * energy_loss + 0.0 * gj_loss = energy_loss
#
# =============================================================================


def save_dataset_to_xlsx(train_dataset, config, output_path: str = None) -> str:
    """
    Save the prepared dataset with all derived features to an xlsx file for inspection.

    Exports the full dataframe including engineered features, with a 'split' column
    indicating train/val/test membership for each row. Includes a second sheet
    with per-feature statistics.

    Args:
        train_dataset: AtomicDataset (train subset) — provides df, feature_columns, target_column
        config: Configuration object
        output_path: Path for the xlsx file. Defaults to data/<elements>_dataset_inspection.xlsx

    Returns:
        Path to the saved xlsx file
    """
    df = train_dataset.df.copy()

    # Attach split labels from the saved split-index file
    split_file = train_dataset._get_split_file_path()
    if os.path.exists(split_file):
        with open(split_file, 'r') as f:
            splits = json.load(f)
        split_map = (
            {idx: 'train' for idx in splits.get('train', [])} |
            {idx: 'val'   for idx in splits.get('val',   [])} |
            {idx: 'test'  for idx in splits.get('test',  [])}
        )
        df['split'] = df.index.map(split_map).fillna('unknown')
    else:
        df['split'] = 'unknown'

    # Determine output path
    if output_path is None:
        if hasattr(config.dataset, 'elements') and config.dataset.elements:
            elements_str = '_'.join(sorted(config.dataset.elements))
        else:
            elements_str = df['Element'].iloc[0] if 'Element' in df.columns else 'dataset'
        data_dir = config.dataset.get('data_dir', 'data')
        output_path = os.path.join(data_dir, f"{elements_str}_dataset_inspection.xlsx")

    # Column order: bookkeeping first, then model features, then everything else
    priority = ['split', 'Element', train_dataset.target_column]
    feature_cols = train_dataset.feature_columns
    other_cols = [c for c in df.columns if c not in priority and c not in feature_cols]
    ordered = (
        [c for c in priority if c in df.columns] +
        [c for c in feature_cols if c in df.columns] +
        [c for c in other_cols if c in df.columns]
    )
    df = df[ordered]

    # Feature statistics for the summary sheet
    feature_summary = pd.DataFrame({
        'Feature': feature_cols,
        'Min':       [df[f].min()   if f in df.columns else None for f in feature_cols],
        'Max':       [df[f].max()   if f in df.columns else None for f in feature_cols],
        'Mean':      [df[f].mean()  if f in df.columns else None for f in feature_cols],
        'Std':       [df[f].std()   if f in df.columns else None for f in feature_cols],
        'NaN_count': [df[f].isna().sum() if f in df.columns else None for f in feature_cols],
    })

    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Dataset', index=True)
            feature_summary.to_excel(writer, sheet_name='Feature Summary', index=False)
    except ImportError:
        print("  ⚠ openpyxl not installed — falling back to CSV")
        output_path = output_path.replace('.xlsx', '.csv')
        df.to_csv(output_path, index=True)

    print(f"\n  ✓ Dataset saved to: {output_path}")
    print(f"    {len(df)} rows × {len(df.columns)} columns | "
          f"{len(feature_cols)} model features")
    return output_path


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

    print("\n" + "=" * 60)
    print("FEATURE VERIFICATION")
    print("=" * 60)

    # Check first sample
    X_first = train_dataset.X[0]
    print(f"First training sample features: {X_first}")
    print(f"Feature names: {train_dataset.get_feature_names()}")
    print(f"Number of features: {len(train_dataset.get_feature_names())}")

    # Check feature ranges
    print(f"\nFeature ranges:")
    for i, name in enumerate(train_dataset.get_feature_names()):
        col_data = train_dataset.X[:, i]
        print(f"  {name}: min={col_data.min():.3f}, max={col_data.max():.3f}, mean={col_data.mean():.3f}")

    # Check target range
    print(f"\nTarget ({train_dataset.target_column}):")
    print(f"  min={train_dataset.y.min():.2f}, max={train_dataset.y.max():.2f}, mean={train_dataset.y.mean():.2f}")

    save_dataset_to_xlsx(train_dataset, config)

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
    best_model_path, best_train_metrics, best_val_metrics = train_model(
        config, model, train_loader, val_loader, device, val_dataset
    )

    return best_model_path, best_train_metrics, best_val_metrics
