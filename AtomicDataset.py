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

# NOTE: Binding energy is now pre-computed in preprocess_atomic.py.
# This dict is retained for convert_predictions_to_absolute() in test_model.py
# and for any future multi-element merging utilities. It is no longer used to
# compute binding-energy targets inside this class (those columns are read from
# the rich XLSX directly).
IONIZATION_ENERGIES = {
    # Element: (Z, ionization_energy_cm-1)
    'Li': (3, 43487.114),
    'Na': (11, 41449.451),  # From NIST
    'K': (19, 35009.814),
    'Fe': (26, 63737.70),
    'Co': (27, 63564.6),
    'Ni': (28, 61619.77),
    'Rb': (37, 33690.81),
    'Cs': (55, 31406.467),
    'Fr': (87, 32848.872)
}
# Alkali metals: the Rydberg features are physically meaningful (1 valence electron)
ALKALI_METALS = {'Li', 'Na', 'K', 'Rb', 'Cs', 'Fr'}
R_INF = 109737.316  # Rydberg constant in cm-1


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
            'target_feature': ds.get('target_feature', None),
            'feature_groups': fg_clean,
            'force_include_features': list(ds.get('force_include_features', []) or []),
            'force_exclude_features': list(ds.get('force_exclude_features', []) or []),
            'normalize_features': ds.normalize_features,
            'normalize_target': ds.normalize_target,
            'multitask_gj': self.config.training.get('multitask_gj', False),
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

        No CSV loading, no feature computation, no Element-column injection.

        Returns:
            DataFrame with all pre-computed columns.
        """
        ds = self.config.dataset

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
            for path in files:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Rich feature file not found: {path}")
                d = pd.read_excel(path, sheet_name='features')
                print(f"  Loaded {len(d)} rows, {d.shape[1]} columns from {path}")
                all_dfs.append(d)
            df = pd.concat(all_dfs, ignore_index=True)
            print(f"  ✓ Combined {len(files)} files: {len(df)} total configurations")
            return df

        # ---- Single element: one rich feature file ----
        path = ds.rich_feature_file
        if not os.path.exists(path):
            raise FileNotFoundError(f"Rich feature file not found: {path}")
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
        """Return the path to the split indices JSON file.

        If config.dataset.split_file is set, it is used directly. Otherwise a name
        is generated from the element column (backward-compatible fallback):
            data/dataset_split_indices_Co.json          (NIST)
            data/dataset_split_indices_Co_kurucz.json   (Kurucz)
        """
        # Explicit config path takes precedence (new behaviour)
        split_file = self.config.dataset.get('split_file', None)
        if split_file:
            return split_file

        # ---- Fallback: generate from the element column ----
        source = self.config.dataset.get('dataset_source', 'nist')
        source_suffix = f'_{source}' if source != 'nist' else ''

        if hasattr(self.config.dataset, 'elements') and \
                self.config.dataset.elements and len(self.config.dataset.elements) > 1:
            # Multi-element: use combined name
            elements_str = '_'.join(sorted(self.config.dataset.elements))
            split_file = f"dataset_split_indices_{elements_str}{source_suffix}.json"
        elif 'Element' in self.df.columns:
            # Single element: element-specific split file
            element = self.df['Element'].iloc[0]
            split_file = f"dataset_split_indices_{element}{source_suffix}.json"
        else:
            split_file = f"dataset_split_indices_dataset{source_suffix}.json"

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
        # Determine split file name
        split_file = self._get_split_file_path()
        print(f"\nData split file: {split_file}")

        # Load existing splits if available
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                splits = json.load(f)
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

    def _create_stratified_splits(self):
        """
        Create stratified splits by element × energy quantile bin.

        For each element, energy levels are divided into n_bins quantile bins
        (equal sample count per bin). The stratum label is 'ELEMENT_binN'.
        This guarantees that both rare low-energy states AND the dense
        high-energy cluster are proportionally represented in every split.

        Works for single-element datasets too: the 'element' part of the
        stratum label is constant, so stratification is purely by energy bin.

        Config key: dataset.stratify_energy_bins (default: 5, set 0 to disable)
        """
        from sklearn.model_selection import train_test_split

        n_bins = int(self.config.dataset.get('stratify_energy_bins', 5))
        raw_level_col = self.config.dataset.target_feature  # original level column

        # ----------------------------------------------------------------
        # Build stratum labels: 'ELEMENT_bin0' .. 'ELEMENT_bin{n_bins-1}'
        # ----------------------------------------------------------------
        strata = []
        for idx, row in self.df.iterrows():
            element = row.get('Element', 'X')

            if n_bins <= 1:
                # No energy stratification — stratify by element only
                strata.append(element)
                continue

            level = row.get(raw_level_col, np.nan)
            try:
                level = float(level)
            except (TypeError, ValueError):
                level = np.nan

            if np.isnan(level):
                strata.append(f"{element}_bin0")
                continue

            # Compute quantile bin within this element's levels
            E_ion = IONIZATION_ENERGIES.get(element, (None, np.nan))[1]
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

            strata.append(f"{element}_bin{bin_label}")

        strata = np.array(strata)

        # ----------------------------------------------------------------
        # Check minimum bin size — sklearn needs >= 2 per stratum per split
        # If any stratum has < 4 samples (can't survive two splits), merge
        # it into the nearest stratum by renaming it.
        # ----------------------------------------------------------------
        from collections import Counter
        counts = Counter(strata)
        MIN_STRATUM_SIZE = 4
        for label, count in counts.items():
            if count < MIN_STRATUM_SIZE:
                # Find replacement: same element, adjacent bin number
                element = label.rsplit('_bin', 1)[0]
                bin_num = int(label.rsplit('_bin', 1)[1])
                # Try bin+1, then bin-1
                for alt in [bin_num + 1, bin_num - 1, bin_num + 2]:
                    alt_label = f"{element}_bin{alt}"
                    if alt_label in counts and counts[alt_label] >= MIN_STRATUM_SIZE:
                        strata[strata == label] = alt_label
                        print(f"    ⚠ Merged stratum '{label}' ({count} samples) "
                              f"→ '{alt_label}'")
                        break

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
        Extract the experimental Landé g-factor as the second prediction target.

        Physics: gJ is measured experimentally but is missing for many levels
        (listed only when the term assignment is confident enough). We store the
        raw (unnormalized) values and a boolean mask so the loss function can skip
        unobserved rows — predicting on partial supervision.

        gJ is NOT normalised: it is dimensionless with a small range (~−1 to 4 for
        Co I) that does not benefit from StandardScaler standardisation the way that
        energy (range ~60 000 cm⁻¹) does.

        Reads pre-computed columns from the rich XLSX:
            obs_gj      — experimental gJ (NaN where unobserved)
            has_obs_gj  — 0/1 mask (1 = observed)

        Stores:
            self.y_gj    — np.float32, shape (N,), gJ values; 0.0 where unobserved
            self.gj_mask — np.float32, shape (N,), 1.0 where experimental gJ exists
        """
        if 'obs_gj' not in self.df.columns:
            raise ValueError(
                "multitask_gj=True requires an 'obs_gj' column in the rich feature "
                "file. Ensure preprocess_atomic.py produced it."
            )

        # Raw gJ values for the current subset rows; NaN → 0.0 (mask handles them)
        self.y_gj = self.df.loc[self.indices, 'obs_gj'].fillna(0.0).values.astype(np.float32)

        # Mask: pre-computed has_obs_gj column (1.0 = observed, 0.0 = missing)
        if 'has_obs_gj' in self.df.columns:
            self.gj_mask = self.df.loc[self.indices, 'has_obs_gj'].values.astype(np.float32)
        else:
            # Fallback (should not happen with the rich xlsx): derive from NaN
            self.gj_mask = (~self.df.loc[self.indices, 'obs_gj'].isna()).values.astype(np.float32)

        n_observed = int(self.gj_mask.sum())
        n_total = len(self.gj_mask)
        print(f"  gJ target ({self.subset}): {n_observed} / {n_total} rows have observed gJ "
              f"({100.0 * n_observed / n_total:.1f}%)")

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

    def inverse_transform_target(self, y_normalized: np.ndarray) -> np.ndarray:
        """
        Convert normalized target values back to original scale (cm⁻¹).

        This is used after making predictions to convert the model's output
        (which is in normalized space) back to physical energy levels.
        Undoes, in order:
            1. StandardScaler normalization (if normalize_target=True)
            2. Inverse scaling: E = A / model_output  (if use_inverse_target=True)

        Args:
            y_normalized: Normalized target values

        Returns:
            Target values in original scale (cm⁻¹)
        """
        y = y_normalized.copy()
        # Step 1: undo StandardScaler
        if self.scaler_target is not None:
            y = self.scaler_target.inverse_transform(y_normalized)

        # Step 2: undo A / E_target  →  E_target = A / y
        if self.config.dataset.get('use_inverse_target', False):
            A = self.config.dataset.get('inverse_target_scale', 100000)
            # Clip to avoid division by zero from a near-zero model output
            clipped = np.clip(y, a_min=1e-6, a_max=None)
            # Warn if any values were actually clipped (signals a poorly predicted sample)
            n_clipped = np.sum(y < 1e-6)
            if n_clipped > 0:
                print(f"  ⚠️  WARNING: {n_clipped} predictions clipped before inversion "
                      f"(model output near zero → unphysically large energy). "
                      f"Min raw value: {y.min():.6f}")
            y = A / clipped

        if self.config.dataset.get('use_log_target', False):
            # RuntimeWarning: overflow encountered in exp
            y = np.exp(y)

        return y

    def inverse_transform_gj(self, y: np.ndarray) -> np.ndarray:
        """
        Identity transform for the gJ target — provided for API consistency.

        Unlike the energy target, gJ is stored and predicted in raw (dimensionless)
        units without any StandardScaler normalisation, so no inverse operation is
        needed. This method exists so that calling code can treat both outputs
        symmetrically.

        Args:
            y: gJ predictions from the model, any shape

        Returns:
            y unchanged
        """
        return y

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
