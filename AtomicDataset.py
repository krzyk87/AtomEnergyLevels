"""
AtomicDataset.py

PyTorch Dataset for atomic energy level prediction.

NEW ARCHITECTURE (as of 2026-06):
Feature engineering has been moved to preprocess_atomic.py, which produces
a rich XLSX file with all features pre-computed. This class now:
  1. Loads the pre-computed XLSX (no feature computation)
  2. Selects feature groups via config (feature_groups section)
  3. Splits data into train/val/test (saved to JSON split file)
  4. Scales features and target (StandardScaler on train set only)
  5. Supports multi-task gJ prediction (multitask_gj config flag)

To add new features: modify preprocess_atomic.py and re-run it.
Then add the new column name to FEATURE_GROUPS in this file.

Author: Aga
For: Physics project on neural network prediction of atomic energy levels
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from typing import Optional
import json
import os
import re

# Per-element physical constants now live in the master table data/element_constants.xlsx
# (see element_data.py). The dataset itself looks up ionization energies through
# _ionization_energy_map / _resolve_ionization_energy. IONIZATION_ENERGIES below is
# kept ONLY as a backward-compatible {symbol: (Z, E_ion)} view for other consumers
# (e.g. visualize.py); it is built from that single source of truth.
R_INF = 109737.316  # Rydberg constant in cm-1


def _ionization_energies_from_table() -> dict:
    """Build {symbol: (Z, E_ion_cm-1)} for neutral species from the master table.

    Returns an empty dict if the table is unavailable, so importing this module
    never fails just because the constants file is missing.
    """
    try:
        from element_data import load_species_table
        table = load_species_table()
        out = {}
        for _, r in table.iterrows():
            if str(r.get('ion_stage', 'I')).strip().upper() != 'I':
                continue  # neutral atoms only, for this legacy symbol-keyed view
            out[str(r['symbol'])] = (int(r['Z']), float(r['ionization_energy_cm-1']))
        return out
    except Exception:
        return {}


IONIZATION_ENERGIES = _ionization_energies_from_table()

# Dataset sources. The rich-feature XLSX and the split JSON are suffixed with the
# active source (dataset.dataset_source) so the NIST and Kurucz datasets — which
# have different rows — never share a file. See preprocess_atomic.py, which writes
# the matching ``_<source>`` rich files.
SOURCE_SUFFIXES = ('nist', 'kurucz')


def with_source_suffix(path, source):
    """
    Return *path* with a ``_<source>`` suffix inserted before its extension.

    Any pre-existing recognised source suffix is stripped first, so the result is
    deterministic whether the configured base path already carries a suffix:

        with_source_suffix('data/Co_features_rich.xlsx', 'kurucz')
            → 'data/Co_features_rich_kurucz.xlsx'
        with_source_suffix('data/dataset_split_indices_Co.json', 'nist')
            → 'data/dataset_split_indices_Co_nist.json'
    """
    if not path:
        return path
    base, ext = os.path.splitext(path)
    for k in SOURCE_SUFFIXES:
        if base.endswith(f'_{k}'):
            base = base[: -(len(k) + 1)]
            break
    return f"{base}_{source}{ext}"


# ---------------------------------------------------------------------------
# Feature group → column mappings.
# Each group name maps to the set of columns that the corresponding
# feature_groups: toggle in config_atomic.yaml will pull in as model inputs.
# These column names must match those produced by preprocess_atomic.py.
# ---------------------------------------------------------------------------
FEATURE_GROUPS = {
    'valence_slots': [
        'val_e1_n', 'val_e1_l', 'val_e2_n', 'val_e2_l', 'val_e3_n', 'val_e3_l',
        'val_e4_n', 'val_e4_l', 'val_e5_n', 'val_e5_l', 'val_e6_n', 'val_e6_l',
        'val_e7_n', 'val_e7_l', 'val_e8_n', 'val_e8_l', 'val_e9_n', 'val_e9_l'
    ],
    'orbital_occupancies': [
        '1s', '2s', '2p', '3s', '3p', '3d', '4s', '4p', '4d', '4f', '5s', '5p', '5d', '6s', '6p'
    ],
    'quantum_numbers': [
        'result_S', 'result_L', 'J', 'parity_computed', 'term_known'
    ],
    'component_terms': [
        'comp1_S', 'comp1_L', 'comp2_S', 'comp2_L', 'comp3_S', 'comp3_L',
        'subres_S', 'subres_L', 'n_components', 'has_subresultant'
    ],
    'angular_momentum': [
        'J_sq', 'L_sq', 'S_sq', 'lande_so_term'
    ],
    'd_electron': [
        'n_3d', 'd_holes', 'd_from_half', 'is_half_filled'
    ],
    'screening': [
        'Z_eff', 'Z_eff_sq'
    ],
    'spin_orbit': [
        'zeta_3d', 'E_so_estimate'
    ],
    'rydberg': [
        'n_star', 'rydberg_prediction', 'one_over_nstar_sq'
    ],
    'gj_features': [
        'calc_gj',     # semi-empirical (Cowan code), circular with energy target
        # lande_g_theoretical ← REMOVED: LS formula is invalid for Co (mixed coupling regime)
        # has_lande_theoretical ← REMOVED: redundant with term_known
    ],
    'eigenvalue': [
        'EIGENVALUE', 'delta_e_theory'
    ],
    'atomic_constants': [
        'Z', 'A', 'proton_number', 'neutron_number'
    ],
    'valence_summary': [
        'valence_electrons', 'total_electrons', 'max_principal_n'
    ],
}


# ---------------------------------------------------------------------------
# Target inversion: map the selected target_feature back to an absolute energy
# level (OBS.LEVEL, cm^-1) for reporting / metrics / predictions.
# ---------------------------------------------------------------------------
RAW_LEVEL_COL = 'OBS.LEVEL'                  # absolute experimental level (cm^-1)
BINDING_COL = 'Binding_Energy_cm-1'          # E_ion - OBS.LEVEL (>=0)
INVERSE_BINDING_COL = 'Inverse_Binding_Energy_cm-1'

# How each target_feature column relates to the binding energy, and thus how a
# model prediction in that space is inverted back to an absolute level.
TARGET_KIND_BY_COLUMN = {
    'OBS.LEVEL':                   'raw',      # already a level → identity
    'Binding_Energy_cm-1':         'binding',  # level = E_ion - y
    'Log_Binding_Energy_cm-1':     'log',      # level = E_ion - exp(y)
    'Inverse_Binding_Energy_cm-1': 'inverse',  # level = E_ion - scale/y
}


class AtomicDataset(Dataset):
    """
    PyTorch Dataset for atomic energy level prediction.

    Loads a pre-computed rich feature XLSX (produced by preprocess_atomic.py),
    selects feature groups by config, splits into train/val/test, and scales.

    Args:
        config: Configuration object containing dataset parameters
        subset: Which data split to use ('train', 'val', or 'test')
        scaler_features: Pre-fitted StandardScaler for features (used for val/test)
        scaler_target: Pre-fitted StandardScaler for target (used for val/test)
    """
    # Class-level cache: shared across all instances
    _data_cache = {}
    _cache_key_counter = 0

    def __init__(
        self,
        config,
        subset: str = 'train',
        scaler_features: Optional[StandardScaler] = None,
        scaler_target: Optional[StandardScaler] = None
    ):
        """
        Initialize the dataset.

        For the training set (subset='train'), this will:
        - Load the pre-computed rich feature file
        - Select feature columns from the configured feature groups
        - Create train/val/test splits
        - Fit normalization scalers on training data

        For validation/test sets, scalers from training must be provided
        to ensure consistent normalization.
        """
        self.config = config
        self.subset = subset

        # Generate cache key based on data configuration
        cache_key = self._generate_cache_key()

        # Check if data already processed.
        # Training always reprocesses so that config changes between sequential
        # experiment runs take effect and the cache is refreshed for the val/test
        # datasets of the same run.
        if cache_key in AtomicDataset._data_cache and subset != 'train':
            print(f"\n{'=' * 60}")
            print(f"Loading CACHED data for {subset} set")
            print(f"{'=' * 60}")

            cached = AtomicDataset._data_cache[cache_key]
            self.df = cached['df']
            self.feature_columns = cached['feature_columns']
            self.target_column = cached['target_column']

            print(f"  ✓ Using preprocessed data: {len(self.df)} total configurations")

        else:
            print(f"\n{'=' * 60}")
            print(f"LOADING PRE-COMPUTED FEATURE FILE")
            print(f"{'=' * 60}")

            # 1. Load the rich feature file (no on-the-fly feature computation)
            self.df = self._load_data()

            print(f"\nLoaded {len(self.df)} atomic energy levels")

            # 2. Validate term symbols (uses rich-xlsx column names)
            self.validate_term_symbol()

            # 3. Target column is selected directly by name. All target variants
            #    (raw / binding / log / inverse) are pre-computed in the xlsx, so
            #    there is no target transformation step here anymore.
            self.target_column = config.dataset.target_feature
            if self.target_column not in self.df.columns:
                raise ValueError(
                    f"target_feature '{self.target_column}' not found in the rich "
                    f"feature file. Available energy columns include: "
                    f"{[c for c in self.df.columns if 'LEVEL' in c.upper() or 'Energy' in c or 'EIGEN' in c]}"
                )

            # 4. Select feature columns from the configured feature groups
            self.feature_columns = self._get_feature_columns()

            # 5. Handle missing values (fill or drop) — unchanged.
            #    NOTE: this runs after _get_feature_columns because it needs
            #    self.feature_columns to know which columns to fill/drop.
            self._handle_missing_values()

            # Cache the preprocessed data
            AtomicDataset._data_cache[cache_key] = {
                'df': self.df.copy(),
                'feature_columns': self.feature_columns,
                'target_column': self.target_column
            }

            print(f"\n✓ Data loading complete and cached")

        # Split data into train/validation/test sets
        self._create_splits()

        # Extract features (X) and target (y) for this subset
        self.X = self.df.loc[self.indices, self.feature_columns].values
        self.y = self.df.loc[self.indices, self.target_column].values.reshape(-1, 1)

        # Resolve how to invert the chosen target back to an absolute level, and
        # recover E_ion from the rich file, so predictions / metrics are reported
        # in physical cm^-1 regardless of which target_feature was selected.
        self._setup_target_inversion()

        # Normalize features and target if requested
        if config.dataset.normalize_features:
            if subset == 'train':
                self.scaler_features = StandardScaler()
                self.X = self.scaler_features.fit_transform(self.X)
            else:
                if scaler_features is None:
                    raise ValueError(f"scaler_features must be provided for subset='{subset}'")
                self.scaler_features = scaler_features
                self.X = self.scaler_features.transform(self.X)
        else:
            self.scaler_features = None

        if config.dataset.normalize_target:
            if subset == 'train':
                self.scaler_target = StandardScaler()
                self.y = self.scaler_target.fit_transform(self.y)
            else:
                if scaler_target is None:
                    raise ValueError(f"scaler_target must be provided for subset='{subset}'")
                self.scaler_target = scaler_target
                self.y = self.scaler_target.transform(self.y)
        else:
            self.scaler_target = None

        # Compute sample weights if requested
        if self.config.dataset.get('use_sample_weights', False):
            self.sample_weights = self._compute_sample_weights()
        else:
            self.sample_weights = None

        # Prepare gJ second target for multi-task learning.
        # Must run AFTER _create_splits() (needs self.indices) and AFTER scaler logic
        # (gJ is not normalised — stored in raw units).
        if config.training.get('multitask_gj', False):
            self._prepare_gj_target()
        else:
            self.y_gj = None
            self.gj_mask = None

        print(f"{subset.capitalize()} set: {len(self)} samples, {self.X.shape[1]} features")

    def _generate_cache_key(self) -> str:
        """
        Generate unique cache key based on data configuration.

        Same configuration = same cache key = reuse processed data. The key now
        reflects the rich-feature-file architecture: the source file(s), target
        column, feature-group toggles, normalization flags, multitask flag and
        the zero-variance filter setting.
        """
        import hashlib
        import json

        ds = self.config.dataset

        # feature_groups → plain sorted dict of bools (OmegaConf-safe)
        fg = ds.get('feature_groups', {}) or {}
        fg_clean = {k: bool(fg.get(k)) for k in sorted(fg.keys())}

        key_dict = {
            'rich_feature_file': ds.get('rich_feature_file', None),
            'rich_feature_files': list(ds.get('rich_feature_files', []) or []),
            # Source matters: NIST and Kurucz load different rich files / row sets,
            # so they must never share a cache entry even with identical settings.
            'dataset_source': ds.get('dataset_source', 'nist'),
            'target_feature': ds.get('target_feature', None),
            'feature_groups': fg_clean,
            'force_include_features': list(ds.get('force_include_features', []) or []),
            'force_exclude_features': list(ds.get('force_exclude_features', []) or []),
            'normalize_features': ds.normalize_features,
            'normalize_target': ds.normalize_target,
            'multitask_gj': self.config.training.get('multitask_gj', False),
            'gj_target_mode': self.config.training.get('gj_target_mode', 'raw'),
            'drop_zero_variance_features': ds.get('drop_zero_variance_features', True),
        }

        # Convert to deterministic string (default=str handles OmegaConf scalars)
        key_str = json.dumps(key_dict, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()

    @classmethod
    def clear_cache(cls):
        """Clear the data cache (useful for testing or memory management)."""
        cls._data_cache = {}
        print("Data cache cleared")

    def _load_data(self) -> pd.DataFrame:
        """
        Load the pre-computed rich feature file(s).

        Reads the 'features' sheet of the rich XLSX produced by
        preprocess_atomic.py. Supports a single file (rich_feature_file) or a
        list (rich_feature_files) which are concatenated; the Element column is
        already present in each xlsx (set by preprocess_atomic.py).

        The configured path is treated as a BASE name: the active dataset source
        (dataset.dataset_source) is appended as a ``_<source>`` suffix so the
        NIST and Kurucz rich files are loaded from separate workbooks, matching
        the ``_<source>`` files written by preprocess_atomic.py.

        No CSV loading, no feature computation, no Element-column injection.

        Returns:
            DataFrame with all pre-computed columns.
        """
        ds = self.config.dataset
        source = ds.get('dataset_source', 'nist')

        # ---- Backward-compatibility guard (Task 9) ----
        # Old configs used 'data_file' / 'elements' to drive CSV loading + on-the-fly
        # feature engineering. That path no longer exists. Give a clear error if a
        # legacy config is supplied without the new rich-feature-file key.
        has_rich = ('rich_feature_file' in ds) or ('rich_feature_files' in ds)
        has_legacy = ('data_file' in ds) or ('elements' in ds)
        if has_legacy and not has_rich:
            raise ValueError(
                "AtomicDataset now requires a pre-computed rich feature file. "
                "Run preprocess_atomic.py first to generate it, then set "
                "'rich_feature_file' in config_atomic.yaml under dataset:. "
                "Old config keys 'data_file' and 'elements' are no longer supported."
            )
        if not has_rich:
            raise ValueError(
                "config.dataset must define 'rich_feature_file' (single element) or "
                "'rich_feature_files' (list, multi-element). Run preprocess_atomic.py "
                "to generate the rich feature XLSX first."
            )

        # ---- Multi-element: list of rich feature files ----
        files = ds.get('rich_feature_files', None)
        if files:
            all_dfs = []
            for raw_path in files:
                path = with_source_suffix(raw_path, source)  # add _<source> suffix
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Rich feature file not found: {path}")
                d = pd.read_excel(path, sheet_name='features')
                print(f"  Loaded {len(d)} rows, {d.shape[1]} columns from {path}")
                all_dfs.append(d)
            df = pd.concat(all_dfs, ignore_index=True)
            print(f"  ✓ Combined {len(files)} files: {len(df)} total configurations")
            return df

        # ---- Single element: one rich feature file ----
        path = with_source_suffix(ds.rich_feature_file, source)  # add _<source> suffix
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Rich feature file not found: {path} (source='{source}'). "
                f"Run: python preprocess_atomic.py --source {source}"
            )
        df = pd.read_excel(path, sheet_name='features')
        print(f"  Loaded {len(df)} rows, {df.shape[1]} columns from {path}")
        return df

    # -------------------------------------------------------------------------
    # REMOVED METHODS (feature computation moved to preprocess_atomic.py)
    # -------------------------------------------------------------------------
    # The following methods were removed in the rich-feature-file refactor.
    # Their logic now lives in preprocess_atomic.py and is baked into the XLSX:
    #
    #   _extract_element_from_filename()  → Element column is written by
    #                                       preprocess_atomic.py.
    #   _encode_valence_electrons()       → val_e1_n..val_e9_l columns (now
    #                                       outermost-first) come from the XLSX.
    #   _add_derived_features()           → valence_electrons / total_electrons /
    #                                       max_principal_n etc. are in the XLSX.
    #   _add_binding_energy_target()      → Binding_Energy_cm-1 is in the XLSX.
    #   _add_inverse_target()             → Inverse_Binding_Energy_cm-1 in XLSX.
    #   _add_log_target()                 → Log_Binding_Energy_cm-1 in the XLSX.
    #   _add_rydberg_features()           → n_star / rydberg_prediction /
    #                                       one_over_nstar_sq in the XLSX
    #                                       (zeros for transition metals).
    #   _add_transition_metal_features()  → J_sq, L_sq, S_sq, lande_so_term,
    #                                       n_3d, Z_eff, E_so_estimate, ... in XLSX.
    #   _add_theoretical_lande_g()        → lande_g_theoretical /
    #                                       has_lande_theoretical in the XLSX.
    #   _impute_SL_from_lande_g()         → handled (if desired) in preprocessing.
    #
    # To change or add a feature: edit preprocess_atomic.py, re-run it, then add
    # the new column name to FEATURE_GROUPS at the top of this file.
    # -------------------------------------------------------------------------

    def _compute_sample_weights(self):
        """
        Compute sample weights based on energy distribution.

        Gives higher weight to underrepresented energy ranges to address
        class imbalance (e.g., few low-energy states, many high-energy states).

        Returns:
            Array of weights for each sample
        """
        # Resolve whichever target column is currently active
        target_col = self.target_column
        energies = self.df.loc[self.indices, target_col].values

        # Choose binning strategy
        weight_strategy = self.config.dataset.get('weight_strategy', 'energy_bins')

        if weight_strategy == 'energy_bins':
            # Bin energies and weight by inverse frequency
            n_bins = self.config.dataset.get('n_energy_bins', 10)
            weights = self._compute_bin_weights(energies, n_bins)

        elif weight_strategy == 'distance_to_ground':
            # Weight by distance from ground state (emphasize low energies)
            weights = self._compute_distance_weights(energies)

        elif weight_strategy == 'kde':
            # Kernel density estimation (smoothest)
            weights = self._compute_kde_weights(energies)

        else:
            raise ValueError(f"Unknown weight_strategy: {weight_strategy}")

        # Normalize weights to average 1.0 (keeps loss scale similar)
        weights = weights / weights.mean()

        print(f"\n  Sample weighting:")
        print(f"    Strategy: {weight_strategy}")
        print(f"    Weight range: {weights.min():.2f} to {weights.max():.2f}")
        print(f"    Avg weight: {weights.mean():.2f}")

        return weights

    def _compute_bin_weights(self, energies: np.ndarray, n_bins: int):
        """
        Compute weights using energy binning.

        Divides energy range into bins and assigns weight = 1/count_in_bin
        """
        # Create bins
        bins = np.linspace(energies.min(), energies.max(), n_bins + 1)
        bin_indices = np.digitize(energies, bins) - 1

        # Count samples per bin
        bin_counts = np.bincount(bin_indices, minlength=n_bins)

        # Avoid division by zero
        bin_counts = np.maximum(bin_counts, 1)

        # Compute weights: inverse frequency
        weights = 1.0 / bin_counts[bin_indices]

        # Report distribution
        print(f"    Energy bins: {n_bins}")
        for i in range(n_bins):
            if bin_counts[i] > 0:
                energy_range = f"{bins[i]:.0f}-{bins[i + 1]:.0f}"
                avg_weight = 1.0 / bin_counts[i]
                print(f"      Bin {i + 1} ({energy_range} cm⁻¹): {bin_counts[i]} samples, weight={avg_weight:.2f}")

        return weights

    def _compute_distance_weights(self, energies: np.ndarray):
        """
        Weight by distance from minimum energy.

        Emphasizes low-energy states (near ground state) over high-energy states.
        """
        min_energy = energies.min()
        distances = energies - min_energy

        # Weight inversely proportional to distance from ground
        # Add small constant to avoid division by zero
        epsilon = (energies.max() - energies.min()) * 0.01
        weights = 1.0 / (distances + epsilon)

        return weights

    def _compute_kde_weights(self, energies: np.ndarray):
        """
        Kernel Density Estimation for smooth weighting.

        Uses Gaussian KDE to estimate density, then weights = 1/density
        """
        from scipy.stats import gaussian_kde

        # Fit KDE
        kde = gaussian_kde(energies)

        # Evaluate density at each point
        densities = kde(energies)

        # Weight = inverse density
        weights = 1.0 / densities

        return weights

    def get_sample_weight(self, idx: int) -> float:
        """Get weight for a specific sample index."""
        if self.sample_weights is None:
            return 1.0
        return self.sample_weights[idx]

    def _get_train_indices_for_variance_check(self):
        """
        Return the training-row indices used for the zero-variance feature filter.

        Loads the split file (same path logic as _create_splits / _get_split_file_path).
        If the split file exists, returns its 'train' indices so that zero-variance
        is evaluated on the TRAINING distribution only (no leakage from val/test into
        feature selection). If no split file exists yet (first run), returns ALL
        DataFrame indices as a safe fallback.
        """
        split_file = self._get_split_file_path()
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                splits = json.load(f)
            return splits.get('train', list(self.df.index))
        return list(self.df.index)

    def _get_feature_columns(self) -> list:
        """
        Select model-input columns from the configured feature groups.

        Reads config.dataset.feature_groups (a dict of group_name → bool). For each
        enabled group, the columns in FEATURE_GROUPS[group_name] that exist in the
        loaded DataFrame are added. force_include_features / force_exclude_features
        then add/remove individual columns, and a zero-variance filter (evaluated on
        the TRAIN indices only) drops uninformative constant columns.

        Returns:
            Ordered, de-duplicated list of feature column names.
        """
        # Ensure 'gj_features' is included if 'multitask_gj' is enabled and excluded otherwise.
        if self.config.training.get('multitask_gj', False):
            self.config.dataset.feature_groups['gj_features'] = True
        elif self.config.dataset.feature_groups.get('gj_features', False):
            raise ValueError(
                "gj_features=true requires multitask_gj=true. "
                "Using calc_gJ as input without predicting obs_gJ introduces "
                "circularity with the Cowan semi-empirical calculation. "
                "Either enable multitask_gj or set gj_features=false."
            )

        features = []
        groups_config = self.config.dataset.get('feature_groups', {})

        # ---- Group-based selection ----
        for group_name, columns in FEATURE_GROUPS.items():
            include = groups_config.get(group_name, False)
            if include:
                available = [c for c in columns if c in self.df.columns]
                missing = [c for c in columns if c not in self.df.columns]
                features.extend(available)
                print(f"  ✓ {group_name}: {len(available)} features added")
                if missing:
                    print(f"    ⚠ Not in xlsx: {missing}")

        # ---- force_include / force_exclude ----
        for col in self.config.dataset.get('force_include_features', []):
            if col in self.df.columns and col not in features:
                features.append(col)
                print(f"  ✓ Force-included: {col}")

        for col in self.config.dataset.get('force_exclude_features', []):
            if col in features:
                features.remove(col)
                print(f"  ✗ Force-excluded: {col}")

        # ---- Zero-variance filter (on TRAIN indices only — not full df) ----
        if self.config.dataset.get('drop_zero_variance_features', True):
            train_mask = self.df.index.isin(self._get_train_indices_for_variance_check())
            before = len(features)
            features = [f for f in features
                        if f in self.df.columns
                        and self.df.loc[train_mask, f].nunique() > 1]
            dropped = before - len(features)
            if dropped > 0:
                print(f"  ✗ Dropped {dropped} zero-variance features (checked on train set)")

        # ---- Deduplicate preserving order ----
        features = list(dict.fromkeys(features))
        features = [f for f in features if f in self.df.columns]

        if len(features) == 0:
            raise ValueError("No valid features found. Check feature_groups config.")

        print(f"\n{'=' * 60}")
        print(f"✓ SELECTED {len(features)} TOTAL FEATURES")
        print(f"{'=' * 60}\n")

        return features

    def _handle_missing_values(self):
        """
        Handle missing values in the dataset.

        Options:
        1. Drop rows with missing values (if drop_missing=True)
        2. Fill missing values with a specified value (default: 0.0)
        """
        # Check for missing values in feature columns
        feature_cols = self.feature_columns + [self.target_column]
        missing_counts = self.df[feature_cols].isnull().sum()

        if missing_counts.sum() > 0:
            print(f"Found missing values:\n{missing_counts[missing_counts > 0]}")

            if self.config.dataset.drop_missing:
                # Drop rows with any missing values
                before = len(self.df)
                self.df = self.df.dropna(subset=feature_cols)
                after = len(self.df)
                print(f"Dropped {before - after} rows with missing values")
            else:
                # Fill missing values with specified value
                fill_value = self.config.dataset.fill_missing_value
                self.df[feature_cols] = self.df[feature_cols].fillna(fill_value)
                print(f"Filled missing values with {fill_value}")

    def _get_split_file_path(self) -> str:
        """Return the path to the split indices JSON file (source-suffixed).

        The NIST and Kurucz datasets contain different rows, so each gets its own
        split file. The active source (dataset.dataset_source) is appended as a
        ``_<source>`` suffix to whichever base name applies:

            data/dataset_split_indices_Co_nist.json     (source = nist)
            data/dataset_split_indices_Co_kurucz.json   (source = kurucz)

        If config.dataset.split_file is set it is used as the base (the suffix is
        injected, replacing any existing source suffix). Otherwise the name is
        generated from the element column (backward-compatible fallback).
        """
        source = self.config.dataset.get('dataset_source', 'nist')

        # Explicit config path takes precedence; the source suffix is injected so
        # nist/kurucz never share a split file even from the same base name.
        split_file = self.config.dataset.get('split_file', None)
        if split_file:
            return with_source_suffix(split_file, source)

        # ---- Fallback: generate from the element column ----
        if hasattr(self.config.dataset, 'elements') and \
                self.config.dataset.elements and len(self.config.dataset.elements) > 1:
            # Multi-element: use combined name
            elements_str = '_'.join(sorted(self.config.dataset.elements))
            split_file = f"dataset_split_indices_{elements_str}_{source}.json"
        elif 'Element' in self.df.columns:
            # Single element: element-specific split file
            element = self.df['Element'].iloc[0]
            split_file = f"dataset_split_indices_{element}_{source}.json"
        else:
            split_file = f"dataset_split_indices_dataset_{source}.json"

        # Add data_dir if specified
        data_dir = self.config.dataset.get('data_dir', None)
        if data_dir:
            split_file = os.path.join(data_dir, split_file)

        return split_file

    def _create_splits(self):
        """
        Create or load train/val/test splits.

        For multi-element data, creates stratified splits to ensure
        each element is represented in all splits.
        """
        # Current stratification scheme (stored in / compared against the file).
        current_scheme = {
            'energy_bins': int(self.config.dataset.get('stratify_energy_bins', 5)),
            'by_gj': self._use_gj_stratification(),
        }

        # Determine split file name
        split_file = self._get_split_file_path()
        print(f"\nData split file: {split_file}")

        # Load existing splits if available
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                splits = json.load(f)
            # Warn (but still use) if the saved split predates the current
            # stratification scheme — delete the file to regenerate it.
            saved_scheme = splits.get('metadata', {}).get('stratification')
            if saved_scheme is not None and dict(saved_scheme) != current_scheme:
                print(f"  ⚠ Existing split was built with a DIFFERENT stratification "
                      f"scheme ({dict(saved_scheme)}) than the current config "
                      f"({current_scheme}). Delete {split_file} to regenerate.")
            self.indices = splits[self.subset]
            print(f"  ✓ Loaded existing {self.subset} split: {len(self.indices)} samples")
            return

        # Create new splits
        print(f"  Creating new splits...")
        np.random.seed(self.config.general.random_seed)

        # Always use energy-stratified splits:
        train_indices, val_indices, test_indices = self._create_stratified_splits()

        # Save splits
        splits = {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices,
            'metadata': {
                'created_date': str(pd.Timestamp.now()),
                'random_seed': self.config.general.random_seed,
                'elements': self.df['Element'].unique().tolist() if 'Element' in self.df.columns else [],
                'stratification': current_scheme,   # energy_bins + by_gj scheme
                'train_size': len(train_indices),
                'val_size': len(val_indices),
                'test_size': len(test_indices),
            }
        }

        os.makedirs(os.path.dirname(split_file) if os.path.dirname(split_file) else '.', exist_ok=True)
        with open(split_file, 'w') as f:
            json.dump(splits, f, indent=2)

        print(f"  ✓ Created and saved splits to {split_file}")
        self._print_split_statistics(train_indices, val_indices, test_indices)

        self.indices = splits[self.subset]

    def _use_gj_stratification(self) -> bool:
        """
        Whether the train/val/test split should also stratify on observed-gJ.

        Only active when ALL of:
          - multi-task gJ learning is enabled (training.multitask_gj), AND
          - dataset.stratify_by_gj is true (default true), AND
          - the 'has_obs_gj' column is present in the loaded data.

        Rationale: the gJ head is supervised only on rows with an observed gJ, so
        balancing has_obs_gj across splits keeps the gJ task trainable and gives
        val/test enough observed rows to evaluate gJ MAE. When multitask is off
        there is no gJ task, so this returns False and the split is identical to
        the previous element × energy-bin behaviour.
        """
        return bool(
            self.config.training.get('multitask_gj', False)
            and self.config.dataset.get('stratify_by_gj', True)
            and 'has_obs_gj' in self.df.columns
        )

    def _create_stratified_splits(self):
        """
        Create stratified splits by element × energy quantile bin, and — when
        multi-task gJ is active — also by observed-gJ presence (has_obs_gj).

        For each element, energy levels are divided into n_bins quantile bins
        (equal sample count per bin). The base stratum label is 'ELEMENT_binN';
        when gJ stratification is on, an observed-gJ suffix is appended:
        'ELEMENT_binN_g0' (gJ missing) / 'ELEMENT_binN_g1' (gJ observed).
        This guarantees rare low-energy states, the dense high-energy cluster,
        AND the observed/missing-gJ rows are proportionally represented in every
        split.

        Works for single-element datasets too: the 'element' part of the
        stratum label is constant, so stratification is purely by energy (× gJ).

        Config keys:
          dataset.stratify_energy_bins  (default 5, set 0 to disable energy bins)
          dataset.stratify_by_gj        (default true; gated on multitask_gj)
        """
        from sklearn.model_selection import train_test_split

        n_bins = int(self.config.dataset.get('stratify_energy_bins', 5))
        # Always bin on the physical absolute level, independent of which
        # target_feature is selected (binding/log/inverse are NOT raw levels).
        raw_level_col = RAW_LEVEL_COL if RAW_LEVEL_COL in self.df.columns \
            else self.config.dataset.target_feature
        use_gj = self._use_gj_stratification()
        if use_gj:
            n_obs = int(self.df['has_obs_gj'].sum())
            print(f"  Stratifying also on observed-gJ (has_obs_gj): "
                  f"{n_obs}/{len(self.df)} rows observed")

        # Per-element ionization energy for log-binding bins (master constants table,
        # with a data-derived fallback). Built once; only needed when energy-binning.
        eion_by_element = self._ionization_energy_map('Element') if n_bins > 1 else {}

        # ----------------------------------------------------------------
        # Build stratum labels: 'ELEMENT_bin{n}' (+ '_g{0|1}' when use_gj)
        # ----------------------------------------------------------------
        strata = []
        for idx, row in self.df.iterrows():
            element = row.get('Element', 'X')
            # Third key: observed-gJ presence (only when gJ stratification is on)
            g_suffix = f"_g{int(row['has_obs_gj'])}" if use_gj else ""

            if n_bins <= 1:
                # No energy stratification — stratify by element (× gJ)
                strata.append(f"{element}{g_suffix}")
                continue

            level = row.get(raw_level_col, np.nan)
            try:
                level = float(level)
            except (TypeError, ValueError):
                level = np.nan

            if np.isnan(level):
                strata.append(f"{element}_bin0{g_suffix}")
                continue

            # Compute quantile bin within this element's levels
            E_ion = eion_by_element.get(str(element), np.nan)
            el_levels_raw = self.df.loc[self.df['Element'] == element, raw_level_col].astype(float)
            el_levels = np.log((E_ion - el_levels_raw).clip(lower=1))  # bin on log(binding)

            # pd.qcut with duplicates='drop' handles ties gracefully
            try:
                bin_label = pd.qcut(
                    el_levels, q=n_bins, labels=False, duplicates='drop'
                ).loc[idx]
                bin_label = int(bin_label) if pd.notna(bin_label) else 0
            except Exception:
                bin_label = 0

            strata.append(f"{element}_bin{bin_label}{g_suffix}")

        strata = np.array(strata)

        # ----------------------------------------------------------------
        # Keep strata populated — sklearn needs >= 2 per stratum per split, and
        # the two-stage 60/20/20 split needs ~4. Merge sparse strata in three
        # scheme-aware steps (works for both 'el_binN' and 'el_binN_gD' labels):
        #   1. Collapse the gJ sub-split of any energy bin that has a sparse
        #      (bin, gj) cell — relabel BOTH gj cells of that bin to the energy-
        #      only base so they actually merge (relaxes gJ balance only where
        #      data is thin; preserves energy stratification).
        #   2. Merge any still-sparse stratum into an adjacent energy bin.
        #   3. Final safety: fold any residual sparse stratum into the largest
        #      stratum so train_test_split can never fail.
        # ----------------------------------------------------------------
        from collections import Counter
        MIN_STRATUM_SIZE = 4
        # Parses 'el_binN' and 'el_binN_gD'; element-only labels won't match.
        label_re = re.compile(r'^(?P<el>.+)_bin(?P<bin>\d+)(?:_g(?P<g>\d))?$')

        # Step 1 — collapse gJ sub-split where a (bin, gj) cell is sparse.
        if use_gj:
            counts = Counter(strata)
            bin_members = {}   # 'el_binN' base → [composite labels in that bin]
            for label in counts:
                m = label_re.match(label)
                if m and m.group('g') is not None:
                    base = f"{m.group('el')}_bin{m.group('bin')}"
                    bin_members.setdefault(base, []).append(label)
            for base, members in bin_members.items():
                if any(counts[lb] < MIN_STRATUM_SIZE for lb in members):
                    for lb in members:
                        strata[strata == lb] = base   # merge g0 & g1 → base
                    print(f"    ⚠ Collapsed gJ split for '{base}' (sparse cell)")

        # Step 2 — merge any still-sparse stratum into an adjacent energy bin.
        counts = Counter(strata)
        for label in [lb for lb, c in counts.items() if c < MIN_STRATUM_SIZE]:
            m = label_re.match(label)
            if not m:
                continue  # element-only label (no _bin) — nothing to merge into
            el, bin_num = m.group('el'), int(m.group('bin'))
            g_sfx = f"_g{m.group('g')}" if m.group('g') is not None else ""
            merged = False
            for alt in (bin_num + 1, bin_num - 1, bin_num + 2, bin_num - 2):
                # Prefer the same gj suffix; fall back to the energy-only base.
                for cand in (f"{el}_bin{alt}{g_sfx}", f"{el}_bin{alt}"):
                    if counts.get(cand, 0) >= MIN_STRATUM_SIZE:
                        strata[strata == label] = cand
                        counts = Counter(strata)
                        print(f"    ⚠ Merged stratum '{label}' → '{cand}'")
                        merged = True
                        break
                if merged:
                    break

        # Step 3 — final safety net: fold any residual sparse stratum into the
        # largest one (guarantees every stratum can survive both split stages).
        counts = Counter(strata)
        if any(c < MIN_STRATUM_SIZE for c in counts.values()):
            largest = max(counts, key=counts.get)
            for label, c in list(counts.items()):
                if c < MIN_STRATUM_SIZE and label != largest:
                    strata[strata == label] = largest
                    print(f"    ⚠ Merged residual stratum '{label}' ({c}) → '{largest}'")

        all_indices = self.df.index.tolist()
        elements = self.df['Element'].values if 'Element' in self.df.columns else None

        # ----------------------------------------------------------------
        # First split: train vs (val + test)
        # ----------------------------------------------------------------
        train_idx, temp_idx = train_test_split(
            all_indices,
            test_size=(self.config.dataset.split.val + self.config.dataset.split.test),
            stratify=strata,  # strata aligned with all_indices
            random_state=self.config.general.random_seed
        )

        # ----------------------------------------------------------------
        # Second split: val vs test
        # Use integer positions into temp_idx to index strata correctly.
        # ----------------------------------------------------------------
        temp_positions = [all_indices.index(i) for i in temp_idx]  # positional
        strata_temp = strata[temp_positions]

        val_ratio = self.config.dataset.split.val / (
                self.config.dataset.split.val + self.config.dataset.split.test
        )
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=(1 - val_ratio),
            stratify=strata_temp,
            random_state=self.config.general.random_seed
        )

        # Print what the strata look like
        print(f"\n  Stratification: {n_bins} energy bins × element")
        for label in sorted(set(strata)):
            n_train = sum(strata[all_indices.index(i)] == label for i in train_idx)
            n_val = sum(strata[all_indices.index(i)] == label for i in val_idx)
            n_test = sum(strata[all_indices.index(i)] == label for i in test_idx)
            print(f"    {label}: train={n_train}, val={n_val}, test={n_test}")

        return train_idx, val_idx, test_idx

    def _create_random_splits(self):
        """Create simple random splits (for single element)."""
        all_indices = self.df.index.tolist()
        np.random.shuffle(all_indices)

        n_total = len(all_indices)
        n_train = int(n_total * self.config.dataset.split.train)
        n_val = int(n_total * self.config.dataset.split.val)

        train_indices = all_indices[:n_train]
        val_indices = all_indices[n_train:n_train + n_val]
        test_indices = all_indices[n_train + n_val:]

        return train_indices, val_indices, test_indices

    def _print_split_statistics(self, train_idx, val_idx, test_idx):
        """Print statistics about the splits, including per-element breakdown."""
        print(f"\n  Split statistics:")
        print(f"    Train: {len(train_idx)} samples ({len(train_idx) / len(self.df) * 100:.1f}%)")
        print(f"    Val:   {len(val_idx)} samples ({len(val_idx) / len(self.df) * 100:.1f}%)")
        print(f"    Test:  {len(test_idx)} samples ({len(test_idx) / len(self.df) * 100:.1f}%)")

        # Per-element breakdown
        if 'Element' in self.df.columns:
            print(f"\n  Per-element distribution:")
            for element in sorted(self.df['Element'].unique()):
                n_train = sum(self.df.loc[train_idx, 'Element'] == element)
                n_val = sum(self.df.loc[val_idx, 'Element'] == element)
                n_test = sum(self.df.loc[test_idx, 'Element'] == element)
                n_total = sum(self.df['Element'] == element)
                print(f"    {element}: Train={n_train}, Val={n_val}, Test={n_test} (Total={n_total})")

    def _prepare_gj_target(self):
        """
        Extract the Landé g-factor auxiliary target for multi-task learning.

        Two modes, controlled by config.training.gj_target_mode:

        'raw' (default for backward compatibility):
            Target = obs_gj (experimental gJ, NaN where unobserved).
            The model predicts the absolute gJ value directly.

        'residual' (preferred for Co I):
            Target = obs_minus_calc_gj = obs_gj − calc_gj.
            The Cowan code already captures the dominant LS-coupling variance in
            gJ, so the ML head learns only the residual correction. This reduces
            the effective prediction range from ~3 units to ~0.1 units, which
            improves convergence and avoids the model wasting capacity on physics
            already handled by the semi-empirical baseline.
            self.calc_gj is stored so that inverse_transform_gj() can recover
            the original obs_gj scale for evaluation.

        In both modes:
            gJ is NOT normalised — it is dimensionless with a small numeric range
            (~−1 to 4 for Co I) that does not benefit from StandardScaler
            normalisation.

        Reads from the rich XLSX (preprocess_atomic.py output):
            obs_gj           — experimental gJ (NaN where unobserved)
            obs_minus_calc_gj — obs_gj − calc_gj (NaN where obs_gj is absent)
            calc_gj          — Cowan-code gJ (always present)
            has_obs_gj       — 0/1 mask (1 = observed)

        Stores:
            self.y_gj    — np.float32 (N,), target values; 0.0 where unobserved
            self.gj_mask — np.float32 (N,), 1.0 where gJ is observed, else 0.0
            self.calc_gj — np.float32 (N,) or None; Cowan baseline for 'residual'
                           mode inverse transform; None in 'raw' mode
        """
        gj_mode = self.config.training.get('gj_target_mode', 'raw')

        if gj_mode == 'residual':
            target_col = 'obs_minus_calc_gj'
            if target_col not in self.df.columns:
                raise ValueError(
                    "gj_target_mode='residual' requires an 'obs_minus_calc_gj' column "
                    "in the rich feature file. Ensure preprocess_atomic.py produced it."
                )
        else:
            target_col = 'obs_gj'
            if target_col not in self.df.columns:
                raise ValueError(
                    "multitask_gj=True requires an 'obs_gj' column in the rich feature "
                    "file. Ensure preprocess_atomic.py produced it."
                )

        # Target values for current subset; NaN → 0.0 (mask excludes these rows from loss)
        self.y_gj = self.df.loc[self.indices, target_col].fillna(0.0).values.astype(np.float32)

        # Mask: a row is supervised only where the CHOSEN target is actually
        # defined (not NaN). In 'raw' mode this equals has_obs_gj. In 'residual'
        # mode it additionally requires calc_gj — so the NIST source, which has no
        # Cowan calc_gj (obs_minus_calc_gj is all-NaN), correctly contributes NO
        # gJ supervision instead of silently training the head toward 0.
        target_valid = self.df.loc[self.indices, target_col].notna().values
        if 'has_obs_gj' in self.df.columns:
            has_obs = self.df.loc[self.indices, 'has_obs_gj'].values.astype(bool)
        else:
            # Fallback: derive from obs_gj NaN (source of truth for "observed")
            has_obs = (~self.df.loc[self.indices, 'obs_gj'].isna()).values
        self.gj_mask = (target_valid & has_obs).astype(np.float32)

        # Warn loudly if residual-mode supervision collapsed to nothing — this is
        # the expected outcome for NIST (no calc_gj); switch to gj_target_mode:'raw'.
        if gj_mode == 'residual' and self.gj_mask.sum() == 0:
            print("  ⚠ gj_target_mode='residual' yields NO supervised gJ rows for "
                  "this source (no calc_gj baseline — e.g. NIST). "
                  "Use gj_target_mode: 'raw' to predict obs_gj directly.")

        # Residual mode: cache calc_gj for this subset so inverse_transform_gj()
        # can add it back and return predictions in the original obs_gj scale.
        if gj_mode == 'residual' and 'calc_gj' in self.df.columns:
            self.calc_gj = self.df.loc[self.indices, 'calc_gj'].values.astype(np.float32)
        else:
            self.calc_gj = None

        n_observed = int(self.gj_mask.sum())
        n_total = len(self.gj_mask)
        print(f"  gJ target ({self.subset}, mode='{gj_mode}'): '{target_col}' — "
              f"{n_observed} / {n_total} rows observed ({100.0 * n_observed / n_total:.1f}%)")

    def __len__(self) -> int:
        """Return the number of samples in this dataset."""
        return len(self.X)

    def __getitem__(self, idx: int):
        """
        Get a single sample from the dataset.

        Single-task mode (multitask_gj=False, the default):
            Returns: (features, y_energy)

        Multi-task mode (multitask_gj=True):
            Returns: (features, y_energy, y_gj, gj_mask)
            - y_gj:    shape (1,), raw experimental gJ (0.0 where unobserved)
            - gj_mask: shape (1,), float32 — 1.0 if gJ observed, 0.0 if missing
        """
        # Convert numpy arrays to PyTorch tensors
        features = torch.FloatTensor(self.X[idx])
        target = torch.FloatTensor(self.y[idx])   # shape (1,) — normalised energy

        if self.config.training.get('multitask_gj', False):
            # Second target: raw experimental gJ value
            y_gj = torch.FloatTensor([self.y_gj[idx]])            # shape (1,)
            # Mask: 1.0 = observed, 0.0 = missing — used by MultiTaskLoss to skip NaN rows
            gj_mask = torch.FloatTensor([float(self.gj_mask[idx])])  # shape (1,)
            return features, target, y_gj, gj_mask

        return features, target

    def get_feature_names(self) -> list:
        """Return the list of feature column names."""
        return self.feature_columns

    def get_input_dim(self) -> int:
        """Return the number of input features (dimensionality)."""
        return len(self.feature_columns)

    def _ionization_energy_map(self, key_col: str) -> dict:
        """
        Return {key_value: E_ion(cm^-1)} for every distinct value of df[key_col].

        Primary source is the master constants table (element_data), looked up by
        species/symbol. Where the table is missing or a key is absent, E_ion is
        derived from the data — OBS.LEVEL + Binding_Energy_cm-1 is exactly E_ion on
        every unclipped row — so old rich files (and elements not yet in the table)
        keep working.
        """
        df = self.df
        keys = [str(k) for k in df[key_col].astype(str).unique()]
        constants_file = self.config.dataset.get('constants_file', None)

        # Data-derived fallback per key (exact where binding > 0).
        derived = {}
        if RAW_LEVEL_COL in df.columns and BINDING_COL in df.columns:
            valid = df[BINDING_COL] > 0
            if valid.any():
                eion = (df[RAW_LEVEL_COL] + df[BINDING_COL])[valid]
                derived = eion.groupby(df.loc[eion.index, key_col].astype(str)).median().to_dict()

        try:
            from element_data import get_ionization_energy as _gie
        except Exception:
            _gie = None

        out, missing = {}, []
        for k in keys:
            val = np.nan
            if _gie is not None:
                try:
                    val = _gie(k, constants_file) if constants_file else _gie(k)
                except (KeyError, FileNotFoundError):
                    val = np.nan
            if np.isnan(val):
                val = derived.get(k, np.nan)
            if np.isnan(val):
                missing.append(k)
            out[k] = val
        if missing:
            print(f"  ⚠️  ionization energy unavailable for {sorted(set(missing))} "
                  f"(constants table + data fallback); related values may be NaN.")
        return out

    def _resolve_ionization_energy(self) -> np.ndarray:
        """
        Per-row ionization energy (cm^-1) for this subset, aligned to self.indices.

        Keyed by each row's 'species' (preferred) or 'Element' symbol, via
        _ionization_energy_map (master table first, data-derived fallback). For a
        single-species file this is a constant column.
        """
        df = self.df
        key_col = 'species' if 'species' in df.columns else (
            'Element' if 'Element' in df.columns else None)
        if key_col is None:
            return np.full((len(self.indices), 1), np.nan)
        emap = self._ionization_energy_map(key_col)
        keys = df.loc[self.indices, key_col].astype(str).values
        return np.array([emap.get(k, np.nan) for k in keys], dtype=float).reshape(-1, 1)

    def _setup_target_inversion(self):
        """
        Resolve the target *kind* and the ionization energy / inverse-scale so
        inverse_transform_target() can map any target_feature back to an absolute
        energy level (OBS.LEVEL, cm^-1).

        E_ion is read from the master constants table (element_data), keyed by the
        row's species/Element — see _resolve_ionization_energy. If the table or a
        species is unavailable it falls back to deriving E_ion from the data
        (OBS.LEVEL + Binding_Energy_cm-1), so nothing breaks without the table.

        Sets:
            self._target_kind   'raw' | 'binding' | 'log' | 'inverse'
            self.E_ion          (N,1) per-row ionization energies (cm^-1), or None for 'raw'
            self._inverse_scale float A in Inverse_Binding = A / binding
        """
        self._target_kind = TARGET_KIND_BY_COLUMN.get(self.target_column, 'raw')
        if self.target_column not in TARGET_KIND_BY_COLUMN:
            print(f"  ⚠️  target_feature '{self.target_column}' is not a known binding "
                  f"transform; inverse_transform_target() will treat it as a raw level "
                  f"(identity).")

        # Inverse-target scale: from config, refined from data when available.
        self._inverse_scale = float(self.config.dataset.get('inverse_target_scale', 100000.0))
        df = self.df
        if INVERSE_BINDING_COL in df.columns and BINDING_COL in df.columns:
            valid = df[BINDING_COL] > 0
            prod = (df[INVERSE_BINDING_COL] * df[BINDING_COL])[valid].dropna()
            if not prod.empty:
                self._inverse_scale = float(prod.median())

        if self._target_kind == 'raw':
            self.E_ion = None
            return

        self.E_ion = self._resolve_ionization_energy()
        if np.all(np.isnan(self.E_ion)):
            raise ValueError(
                f"Cannot invert target '{self.target_column}' to absolute levels: no "
                f"ionization energy available from the constants table or the data "
                f"(need element_constants.xlsx or OBS.LEVEL+{BINDING_COL} columns)."
            )

    def inverse_transform_target(self, y_normalized: np.ndarray) -> np.ndarray:
        """
        Convert model output back to an absolute energy level in cm⁻¹.

        Undoes, in order:
            1. StandardScaler normalization (if normalize_target=True)
            2. the binding-energy transform implied by target_feature, all the way
               back to an absolute level:
                   raw      → identity                 (already a level)
                   binding  → level = E_ion - y
                   log      → level = E_ion - exp(y)
                   inverse  → level = E_ion - scale/y

        Because targets and predictions are both routed through this function, all
        reported metrics (train/val/test MAE, RMSE, …) and the saved predictions
        come out in physical cm⁻¹ for every target_feature, and inverting the stored
        targets round-trips back to OBS.LEVEL.

        Args:
            y_normalized: model output (or stored target) in the target space.

        Returns:
            Absolute energy levels in cm⁻¹, same shape as the input.
        """
        y = np.asarray(y_normalized, dtype=float).copy()

        # Step 1: undo StandardScaler (only if the target was standardized).
        if self.scaler_target is not None:
            y = self.scaler_target.inverse_transform(y)

        kind = getattr(self, '_target_kind', 'raw')
        if kind == 'raw':
            return y                                    # already an absolute level

        # Step 2: undo the binding-energy transform → binding energy (cm^-1).
        if kind == 'log':
            # Clip the exponent so a wild prediction cannot overflow exp().
            binding = np.exp(np.clip(y, a_min=None, a_max=25.0))
        elif kind == 'inverse':
            n_clipped = int(np.sum(y < 1e-6))
            if n_clipped > 0:
                print(f"  ⚠️  WARNING: {n_clipped} predictions clipped before inverse-target "
                      f"inversion (model output near zero → unphysically large energy).")
            binding = self._inverse_scale / np.clip(y, a_min=1e-6, a_max=None)
        else:  # 'binding'
            binding = y

        # Step 3: binding energy → absolute level: E_level = E_ion - binding.
        return self.E_ion - binding

    def inverse_transform_gj(self, y: np.ndarray,
                              calc_gj: np.ndarray = None) -> np.ndarray:
        """
        Recover obs_gj predictions from model output.

        In 'raw' mode (gj_target_mode='raw'):
            The model predicts obs_gj directly — no inversion needed (identity).

        In 'residual' mode (gj_target_mode='residual'):
            The model predicts obs_minus_calc_gj = obs_gj − calc_gj, so
            recovering obs_gj requires adding the Cowan baseline back:
                obs_gj ≈ y_pred + calc_gj

        Args:
            y:        Model gJ predictions, any shape (batch or full-subset array).
            calc_gj:  Cowan-code baseline values, same shape as y.
                      Required when gj_target_mode='residual'; ignored otherwise.
                      If None in residual mode, self.calc_gj is used as a fallback
                      (works when y covers the entire stored subset).

        Returns:
            np.ndarray: obs_gj predictions in raw (dimensionless) units.
        """
        gj_mode = self.config.training.get('gj_target_mode', 'raw')
        if gj_mode != 'residual':
            return y

        # Residual mode: add Cowan baseline back to recover the obs_gj scale
        if calc_gj is not None:
            return y + calc_gj

        # Fallback: use the stored subset array (only valid for full-subset evaluation)
        if self.calc_gj is not None:
            return y + self.calc_gj

        raise ValueError(
            "inverse_transform_gj: gj_target_mode='residual' but no calc_gj was "
            "provided and self.calc_gj is None. Pass calc_gj explicitly."
        )

    def validate_term_symbol(self):
        """
        Verify that the Term symbol matches result_S, result_L, J values.
        This catches data entry errors.

        Column names reflect the rich-xlsx schema:
            Term_raw   (resultant term symbol)
            result_S   (total spin S of the resultant term)
            result_L   (total orbital momentum L of the resultant term)
        """
        if 'Term_raw' not in self.df.columns:
            print("  ℹ️  No 'Term_raw' column found - skipping validation")
            return

        # Mapping from L quantum number to letter
        L_LETTER_MAP = {0: 'S', 1: 'P', 2: 'D', 3: 'F', 4: 'G', 5: 'H', 6: 'I', 7: 'K'}

        errors = []

        for idx, row in self.df.iterrows():
            term = row['Term_raw']
            S_qn = row['result_S']
            L_qn = row['result_L']
            J = row['J']

            # Missing term (e.g. blanked NIST non-LS entries like '16*', '*'):
            # nothing to validate — these are intentionally recorded as missing.
            if pd.isna(term) or str(term).strip() == '':
                continue

            # Parse term symbol: ²P₃/₂ → multiplicity=2, L_letter='P', J_term=1.5
            # Examples: "2S", "2P*", "2D", "4F", "2[3/2]*"
            # Pattern: (multiplicity)(L_letter)(optional_subscript)(optional_asterisk)
            match = re.match(r'^(\d+)([A-Z]).*?(\*?)$', str(term))

            if not match:
                # Handle bracket notation: "2[3/2]*"
                match_bracket = re.match(r'^(\d+)\[.*?\](\*?)$', str(term))
                if match_bracket:
                    continue  # Skip bracket notation (different convention)
                errors.append(f"Row {idx}: Cannot parse term '{term}'")
                continue

            multiplicity_str, L_letter, asterisk = match.groups()
            multiplicity_term = int(multiplicity_str)

            # Skip rows with unknown S/L (cannot cross-check)
            if pd.isna(S_qn) or pd.isna(L_qn):
                continue

            # Check multiplicity: should equal 2*S + 1
            multiplicity_expected = int(2 * S_qn + 1)
            if multiplicity_term != multiplicity_expected:
                errors.append(
                    f"Row {idx}: Term '{term}' has multiplicity {multiplicity_term}, "
                    f"but result_S={S_qn} implies {multiplicity_expected}"
                )

            # Check L letter
            L_letter_expected = L_LETTER_MAP.get(int(L_qn), '?')
            if L_letter != L_letter_expected:
                errors.append(
                    f"Row {idx}: Term '{term}' has L='{L_letter}', "
                    f"but result_L={L_qn} implies '{L_letter_expected}'"
                )

            # Check parity (asterisk = odd parity)
            if 'parity_flag' in row:
                parity = row['parity_flag']
                has_asterisk = (asterisk == '*')
                # Note: This check depends on how parity is encoded in your data.
                # Co rich-xlsx Term_raw has no '*' (parity is in parity_flag), so this
                # block is informational only and can be adjusted as needed.

        if errors:
            print(f"⚠️  Found {len(errors)} term symbol mismatches:")
            for err in errors[:5]:  # Show first 5
                print(f"   {err}")
        else:
            print(f"✓ Term symbols validated: all consistent with result_S, result_L, J")


# =============================================================================
# TESTING CHECKLIST (verify after implementation)
# =============================================================================
#  [x] Loading rich xlsx produces same row count as input (363 rows for Co)
#  [x] With same feature_groups as old config flags, same features are selected
#  [x] Train/val/test splits load from existing JSON file (not regenerated)
#  [x] Scaler fitted on train only; val/test use train scaler
#  [x] Zero-variance filter uses train indices only
#  [x] multitask_gj: true works: obs_gj is target, has_obs_gj is mask
#  [x] feature_groups: rydberg: true adds rydberg columns (all zeros for Co — ok)
#  [x] force_exclude_features removes a column that was group-included
#  [x] validate_term_symbol() runs without KeyError on new column names
#  [x] Backward compat: old config with 'data_file' raises clear ValueError
#  [x] Cache key differs between feature_groups configs (no stale cache)
# =============================================================================
