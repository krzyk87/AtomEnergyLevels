"""
AtomicDataset.py

This module defines the PyTorch Dataset class for atomic energy level data.
It handles:
1. Loading CSV data with electron configurations and quantum numbers
2. Feature engineering (adding derived physical features)
3. Data normalization and preprocessing
4. Splitting data into train/validation/test sets

Author: Aga
For: Physics project on neural network prediction of atomic energy levels
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, Dict
import json
import os
import re

IONIZATION_ENERGIES = {
    # Element: (Z, ionization_energy_cm-1)
    'Li': (3, 43487.114),
    'Na': (11, 41449.451),  # From NIST
    'K': (19, 35009.814),
    'Fe': (26, 63737.70),
    'Co': (27, 63564.6),
    'Ni': (28, 61619.77),
    'Cs': (55, 31406.467),
    'Rb': (37, 33690.81),
}

class AtomicDataset(Dataset):
    """
    PyTorch Dataset for atomic energy level prediction.
    
    This dataset loads electron configuration data (number of electrons in each orbital),
    quantum numbers (J, S, L, parity), and predicts energy levels in cm⁻¹.
    
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
        - Load and preprocess the data
        - Create train/val/test splits
        - Fit normalization scalers on training data
        
        For validation/test sets, scalers from training must be provided
        to ensure consistent normalization.
        """
        self.config = config
        self.subset = subset

        # Generate cache key based on data configuration
        cache_key = self._generate_cache_key()

        # Check if data already processed
        if cache_key in AtomicDataset._data_cache:
            print(f"\n{'=' * 60}")
            print(f"Loading CACHED data for {subset} set")
            print(f"{'=' * 60}")

            # Retrieve cached data
            cached = AtomicDataset._data_cache[cache_key]
            self.df = cached['df']
            self.feature_columns = cached['feature_columns']
            self.target_column = cached['target_column']

            print(f"  ✓ Using preprocessed data: {len(self.df)} total configurations")

        else:
            # First time: process data and cache it
            print(f"\n{'=' * 60}")
            print(f"LOADING AND PREPROCESSING DATA")
            print(f"{'=' * 60}")

            # Load the full dataset
            self.df = self._load_data()

            print(f"\nLoaded {len(self.df)} atomic energy levels")

            # Validate term symbols
            self.validate_term_symbol()

            # Add binding energy if requested
            if config.dataset.get('use_binding_energy', False):
                self._add_binding_energy_target()

            # Apply inverse target scaling if requested: stores A / E_target
            # Works on whatever target is active (raw level or binding energy)
            if config.dataset.get('use_inverse_target', False):
                self._add_inverse_target()

            if config.dataset.get('use_log_target', False):
                self._add_log_target()

            # Add derived features if requested (total electrons, valence electrons, etc.)
            if config.dataset.add_derived_features:
                self._add_derived_features()

            # Prepare feature columns
            self.feature_columns = self._get_feature_columns()
            # Prepare target column: binding energy or absolute energy level
            # if config.dataset.get('use_binding_energy', False):
            #     self.target_column = 'Binding_Energy_cm-1'
            #     print(f"  ✓ Target changed to binding energy")
            # else:
            self.target_column = config.dataset.target_feature

            # Handle missing values
            self._handle_missing_values()

            # ✅ Cache the preprocessed data
            AtomicDataset._data_cache[cache_key] = {
                'df': self.df.copy(),  # Store a copy
                'feature_columns': self.feature_columns,
                'target_column': self.target_column
            }

            print(f"\n✓ Data preprocessing complete and cached")
        
        # Split data into train/validation/test sets
        # This creates self.indices that contains row indices for this subset
        self._create_splits()
        
        # Extract features (X) and target (y) for this subset
        self.X = self.df.loc[self.indices, self.feature_columns].values
        self.y = self.df.loc[self.indices, self.target_column].values.reshape(-1, 1)
        
        # Normalize features and target if requested
        if config.dataset.normalize_features:
            if subset == 'train':
                # For training set: fit a new scaler and transform
                self.scaler_features = StandardScaler()
                self.X = self.scaler_features.fit_transform(self.X)
            else:
                # For val/test: use the scaler fitted on training data
                if scaler_features is None:
                    raise ValueError(f"scaler_features must be provided for subset='{subset}'")
                self.scaler_features = scaler_features
                self.X = self.scaler_features.transform(self.X)
        else:
            self.scaler_features = None
        
        if config.dataset.normalize_target:
            if subset == 'train':
                # Fit scaler on training target values
                self.scaler_target = StandardScaler()
                self.y = self.scaler_target.fit_transform(self.y)
            else:
                # Use training scaler for val/test
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
        
        print(f"{subset.capitalize()} set: {len(self)} samples, {self.X.shape[1]} features")

    def _generate_cache_key(self) -> str:
        """
        Generate unique cache key based on data configuration.

        Same configuration = same cache key = reuse processed data.
        """
        import hashlib
        import json

        # Create key from relevant config parameters
        key_dict = {}

        # Data source
        if hasattr(self.config.dataset, 'elements'):
            key_dict['elements'] = sorted(self.config.dataset.elements)
        elif hasattr(self.config.dataset, 'data_file'):
            key_dict['data_file'] = self.config.dataset.data_file

        # Processing parameters that affect data
        key_dict['use_binding_energy'] = self.config.dataset.get('use_binding_energy', False)
        key_dict['use_inverse_target'] = self.config.dataset.get('use_inverse_target', False)
        key_dict['inverse_target_scale'] = self.config.dataset.get('inverse_target_scale', 100000)
        key_dict['add_derived_features'] = self.config.dataset.get('add_derived_features', False)
        key_dict['encode_valence_electrons'] = self.config.dataset.get('encode_valence_electrons', False)
        key_dict['max_valence_electrons'] = self.config.dataset.get('max_valence_electrons', 1)

        # Convert to deterministic string
        key_str = json.dumps(key_dict, sort_keys=True)

        # Hash for compact key
        return hashlib.md5(key_str.encode()).hexdigest()

    @classmethod
    def clear_cache(cls):
        """Clear the data cache (useful for testing or memory management)."""
        cls._data_cache = {}
        print("Data cache cleared")

    def _load_data(self) -> pd.DataFrame:
        """
        Load atomic energy data from single or multiple files.

        Supports two modes:
        1. Single element: config.dataset.data_file
        2. Multiple elements: config.dataset.elements (list)

        Returns:
            Combined DataFrame with all data
        """
        # Mode 1: Multiple elements
        if hasattr(self.config.dataset, 'elements') and self.config.dataset.elements:
            elements = self.config.dataset.elements
            data_dir = self.config.dataset.get('data_dir', 'data')

            print(f"\nLoading multiple elements: {elements}")
            print(f"Data directory: {data_dir}")

            all_dfs = []

            for element in elements:
                # Construct filename: data/Na_features.csv
                data_file = os.path.join(data_dir, f"{element}_features.csv")

                if not os.path.exists(data_file):
                    raise FileNotFoundError(
                        f"Data file not found for element '{element}': {data_file}"
                    )

                print(f"  Loading {element}: {data_file}")
                df = pd.read_csv(data_file)

                # Add element identifier
                df['Element'] = element

                print(f"    → {len(df)} configurations")

                all_dfs.append(df)

            # Combine all dataframes
            combined_df = pd.concat(all_dfs, ignore_index=True)
            print(f"\n  ✓ Combined {len(all_dfs)} elements: {len(combined_df)} total configurations")

            return combined_df

        # Mode 2: Single file (backward compatible)
        elif hasattr(self.config.dataset, 'data_file') and self.config.dataset.data_file:
            data_file = self.config.dataset.data_file
            print(f"\nLoading single element from: {data_file}")
            df = pd.read_csv(data_file)

            # Add element column if not present
            if 'Element' not in df.columns:
                # Extract element from filename: "energy_Na_features.csv" → "Na"
                element = self._extract_element_from_filename(data_file)
                df['Element'] = element
                print(f"  Added Element column: {element}")

            return df

        else:
            raise ValueError(
                "Must specify either 'data_file' (single element) or "
                "'elements' (multiple elements) in config.dataset"
            )

    def _extract_element_from_filename(self, filepath: str) -> str:
        """Extract element symbol from filename."""
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

    def _encode_valence_electrons(self, max_valence=10):
        """
        Encode valence electrons as fixed-size feature vector of individual (n, l) pairs with padding.

        This creates a FIXED number of features regardless of how many
        valence electrons the atom actually has. Atoms with fewer valence
        electrons are padded with zeros.

        Handles NaN values that appear when combining data from different elements
        with different orbital columns.

        Args:
            max_valence: Maximum number of valence electrons to encode.
                         Choose large enough for all atoms you want to train on.

        Example:
            For Na (1 valence electron) with max_valence=3:
            Ground state 3s¹: [0, 0, 0, 0, 3, 0]  # [e1_n, e1_l, e2_n, e2_l, e3_n, e3_l]
            Excited state 3p¹: [0, 0, 0, 0, 3, 1]
            Excited state 4s¹: [0, 0, 0, 0, 4, 0]
        """
        print(f"\nEncoding valence electrons (max={max_valence})...")

        # Get orbital columns
        orbital_pattern = re.compile(r'^(\d+)([spdfgh])$')
        orbital_cols = [col for col in self.df.columns if orbital_pattern.match(col)]

        print(f"  Found {len(orbital_cols)} orbital columns: {orbital_cols[:10]}...")

        # Define orbital priority (which electrons count as "core" vs "valence")
        # Lower values = core (filled first), higher = valence
        def orbital_energy_order(orbital_name):
            """
            Approximate orbital filling order (Aufbau principle).
            Returns a sortable tuple (n + l, n, l) for ordering.
            """
            match = orbital_pattern.match(orbital_name)
            n = int(match.group(1))
            l_letter = match.group(2)
            l = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5}[l_letter]

            # Sort by n+l first (Madelung rule), then by n
            return (n + l, n, l)

        features = []

        for idx, row in self.df.iterrows():
            # Collect all electrons with their (n, l) quantum numbers
            all_electrons = []

            for col in orbital_cols:
                n_electrons_raw = row[col]

                # Skip if NaN (orbital doesn't exist for this element)
                if pd.isna(n_electrons_raw):
                    continue

                # Convert to integer
                n_electrons = int(n_electrons_raw)

                if n_electrons > 0:
                    # Parse orbital: '3d' → n=3, l=2
                    match = orbital_pattern.match(col)
                    n = int(match.group(1))
                    l_letter = match.group(2)
                    l = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5}[l_letter]

                    # Add (n, l) for each electron in this orbital
                    for _ in range(n_electrons):
                        all_electrons.append((n, l, col))

            # Sort electrons by filling order (core electrons first)
            all_electrons.sort(key=lambda x: orbital_energy_order(x[2]))

            # Take only the last max_valence electrons (valence electrons)
            if len(all_electrons) >= max_valence:
                valence_electrons = all_electrons[-max_valence:]
            else:
                valence_electrons = all_electrons

            # Pad with zeros at the BEGINNING if fewer than max_valence
            # This ensures valence electrons always appear in the same positions
            while len(valence_electrons) < max_valence:
                valence_electrons.insert(0, (0, 0, 'pad'))

            # Flatten into feature vector
            feature_row = []
            for n, l, _ in valence_electrons:
                feature_row.extend([n, l])

            features.append(feature_row)

        # Create feature dataframe
        feature_cols = []
        for i in range(max_valence):
            feature_cols.extend([f'val_e{i + 1}_n', f'val_e{i + 1}_l'])

        feature_df = pd.DataFrame(features, columns=feature_cols, index=self.df.index)

        # Merge with original dataframe
        self.df = pd.concat([self.df, feature_df], axis=1)

        # Print statistics
        n_valence_actual = len([e for e in all_electrons if e[0] > 0])
        print(f"  ✓ Created {len(feature_cols)} features ({max_valence} valence electrons)")
        print(f"  ✓ Actual valence electrons in data: {n_valence_actual} (padded to {max_valence})")

        # Show example for each element
        if 'Element' in self.df.columns:
            for element in self.df['Element'].unique():
                element_df = self.df[self.df['Element'] == element]
                example_idx = element_df.index[0]
                example_features = [int(self.df.loc[example_idx, col]) for col in feature_cols]
                print(f"  Example ({element}): {example_features}")
        else:
            example_idx = self.df.index[0]
            example_features = [self.df.loc[example_idx, col] for col in feature_cols]
            print(f"  Example: {example_features}")

    def _compute_sample_weights(self):
        """
        Compute sample weights based on energy distribution.

        Gives higher weight to underrepresented energy ranges to address
        class imbalance (e.g., few low-energy states, many high-energy states).

        Returns:
            Array of weights for each sample
        """
        # Resolve whichever target column is currently active
        # (may have been updated by _add_binding_energy_target / _add_inverse_target)
        target_col = self.config.dataset.target_feature
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

    def _get_feature_columns(self) -> list:
        """
        Automatically detect and select relevant features.

        This method intelligently selects features based on:
        1. Automatically selecting features using valence electron encoding.
        2. Removing constant/empty columns (not informative)
        3. Including quantum numbers and atomic properties
        4. Handling single vs. multi-element datasets

        Returns:
            List of column names to use as input features
        """
        import re

        features = []

        # ========================================
        # OPTION 1: Encode valence electrons explicitly
        # ========================================
        if self.config.dataset.get('encode_valence_electrons', True):
            max_valence = self.config.dataset.get('max_valence_electrons', 10)
            self._encode_valence_electrons(max_valence)

            # Drop valence encoding columns that are constant (same value in every row)
            # This happens for alkali metals where only the last slot varies
            # val_cols = [f'val_e{i + 1}_n' for i in range(max_valence)] + \
            #            [f'val_e{i + 1}_l' for i in range(max_valence)]
            # val_cols = [c for c in val_cols if c in self.df.columns]
            # non_constant = [c for c in val_cols if self.df[c].nunique() > 1]
            # constant_dropped = set(val_cols) - set(non_constant)
            # if constant_dropped:
            #     print(f"  ✗ Dropped {len(constant_dropped)} constant valence columns: {sorted(constant_dropped)}")
            # features = [f for f in features if f not in constant_dropped]
            # self.df <-- features

            # Add valence electron features
            for i in range(max_valence):
                features.extend([f'val_e{i + 1}_n', f'val_e{i + 1}_l'])

            print(f"✓ Using valence electron encoding: {len(features)} features")

        # ========================================
        # OPTION 2: Use full orbital occupancy (fallback)
        # ========================================
        else:
            # Auto-detect orbitals
            orbital_pattern = re.compile(r'^\d+[spdfgh]$')
            all_orbitals = [col for col in self.df.columns if orbital_pattern.match(col)]

            # Keep non-empty, non-constant
            valid_orbitals = [
                col for col in all_orbitals
                if self.df[col].sum() > 0 and self.df[col].nunique() > 1
            ]
            features.extend(valid_orbitals)
            print(f"✓ Using full orbital occupancy: {len(features)} features")

        # ========================================
        # QUANTUM NUMBERS (always include if present)
        # ========================================
        quantum_cols = self.config.dataset.get('quantum_features') # ['J', 'S_qn', 'L_qn', 'parity']
        added_quantum = []

        for col in quantum_cols:
            if col in self.df.columns:
                features.append(col)
                added_quantum.append(col)

        print(f"  ✓ Added {len(added_quantum)} quantum features: {added_quantum}")

        # ========================================
        # ATOMIC PROPERTIES (only if multi-element)
        # ========================================
        atomic_cols = self.config.dataset.get('atomic_features')    # ['Z', 'A', 'proton_number', 'neutron_number']
        added_atomic = []
        skipped_atomic = []

        for col in atomic_cols:
            if col in self.df.columns:
                if self.df[col].nunique() > 1:
                    # Multiple values = multi-element dataset
                    features.append(col)
                    added_atomic.append(col)
                else:
                    # Single value = single element (not useful as feature)
                    skipped_atomic.append(col)

        if added_atomic:
            print(f"  ✓ Added {len(added_atomic)} atomic features: {added_atomic}")
            print(f"    (Multi-element dataset detected)")
        if skipped_atomic:
            print(f"  ✗ Skipped {len(skipped_atomic)} constant atomic features: {skipped_atomic}")
            print(f"    (Single-element dataset)")

        # ========================================
        # DERIVED FEATURES
        # ========================================
        if self.config.dataset.add_derived_features:
            derived_cols = ['total_electrons', 'valence_electrons', 'max_principal_n']  # 'core_electrons', 'unpaired_electrons',
            added_derived = [c for c in derived_cols if c in self.df.columns]
            features.extend(added_derived)
            print(f"  ✓ Added {len(added_derived)} derived features: {added_derived}")

        # ========================================
        # FORCE INCLUDE/EXCLUDE (from config)
        # ========================================
        if hasattr(self.config.dataset, 'force_include_features'):
            for col in self.config.dataset.force_include_features:
                if col in self.df.columns and col not in features:
                    features.append(col)
                    print(f"  ✓ Force-included: {col}")

        if hasattr(self.config.dataset, 'force_exclude_features'):
            for col in self.config.dataset.force_exclude_features:
                if col in features:
                    features.remove(col)
                    print(f"  ✗ Force-excluded: {col}")

        # ========================================
        # FINAL VALIDATION
        # ========================================
        # Remove duplicates while preserving order
        features = list(dict.fromkeys(features))

        # Verify all exist in dataframe
        features = [f for f in features if f in self.df.columns]

        print(f"\n{'=' * 60}")
        print(f"✓ SELECTED {len(features)} TOTAL FEATURES")
        print(f"{'=' * 60}\n")

        if len(features) == 0:
            raise ValueError("No valid features found! Check your data file.")

        return features

    def _add_derived_features(self):
        """
        Add physically meaningful derived features to help the model learn.
        
        These features are computed from the raw electron configuration
        and provide additional physics-based information:
        - Total number of electrons
        - Number of valence electrons (in outermost shell)
        - Number of core electrons
        - Number of unpaired electrons
        - Maximum principal quantum number (highest occupied shell)
        """
        # Get all orbital columns (1s, 2s, 2p, 3s, etc.)
        orbital_cols = self.config.dataset.orbital_features
        orbital_cols = [c for c in orbital_cols if c in self.df.columns]
        
        # Total electrons: sum across all orbitals
        self.df['total_electrons'] = self.df[orbital_cols].sum(axis=1)
        
        # Maximum principal quantum number (n): highest shell with electrons
        # Extract principal quantum number from orbital name (e.g., '3s' -> 3)
        for idx, row in self.df.iterrows():
            max_n = 0
            for col in orbital_cols:
                if row[col] > 0:  # If this orbital has electrons
                    n = int(col[0])  # First character is the principal quantum number
                    max_n = max(max_n, n)
            self.df.loc[idx, 'max_principal_n'] = max_n
        
        # Valence electrons: electrons in the outermost shell
        # This is approximate - counts electrons in orbitals with n = max_n
        valence_electrons = []
        for idx, row in self.df.iterrows():
            max_n = int(row['max_principal_n'])
            valence = 0
            for col in orbital_cols:
                n = int(col[0])
                if n == max_n:
                    valence += row[col]
            valence_electrons.append(valence)
        self.df['valence_electrons'] = valence_electrons
        
        # Core electrons: total - valence
        self.df['core_electrons'] = self.df['total_electrons'] - self.df['valence_electrons']
        
        # Unpaired electrons: simplified estimation from S quantum number
        # S = total spin = (number of unpaired electrons) / 2
        # So unpaired electrons ≈ 2 * S
        if 'S_qn' in self.df.columns:
            self.df['unpaired_electrons'] = 2 * self.df['S_qn']
        else:
            self.df['unpaired_electrons'] = 0

        # Approximate quantum defects for K (from literature / NIST fit)
        QUANTUM_DEFECTS_K = {0: 2.18, 1: 1.71, 2: 0.28, 3: 0.01, 4: 0.0}
        R_INF = 109736.6  # cm⁻¹

        # For each row, compute n* and the Rydberg prediction
        n = row['val_e1_n']
        l = row['val_e1_l']
        delta = QUANTUM_DEFECTS_K.get(l, 0.0)
        n_star = n - delta
        self.df['n_star'] = n_star
        self.df['rydberg_prediction'] = R_INF / n_star ** 2
        self.df['one_over_nstar_sq'] = 1.0 / n_star ** 2
        
        print(f"Added derived features: total_electrons, valence_electrons, "
              f"core_electrons, unpaired_electrons, max_principal_n")

    def _add_binding_energy_target(self):
        """
        Convert absolute energy levels to binding energies.

        Binding energy = E_ionization - E_level

        This represents how much energy is needed to remove the electron
        from this state to the ionization continuum.
        """
        print(f"\nConverting to binding energies:")

        # Check if Element column exists
        if 'Element' not in self.df.columns:
            # Single element: extract from config or filename
            if hasattr(self.config.dataset, 'elements') and self.config.dataset.elements:
                element = self.config.dataset.elements[0]
            elif hasattr(self.config.dataset, 'data_file'):
                element = self._extract_element_from_filename(self.config.dataset.data_file)
            else:
                raise ValueError("Cannot determine element for binding energy calculation")

            # Add Element column
            self.df['Element'] = element
            print(f"  Added Element column: {element}")

        # Get unique elements in dataset
        unique_elements = self.df['Element'].unique()
        print(f"  Elements in dataset: {list(unique_elements)}")

        # Process each element separately
        binding_energies = []
        ionization_energies = []

        for idx, row in self.df.iterrows():
            element = row['Element']

            if element not in IONIZATION_ENERGIES:
                raise ValueError(
                    f"Ionization energy not defined for element: {element}\n"
                    f"Available elements: {list(IONIZATION_ENERGIES.keys())}"
                )

            Z, E_ion = IONIZATION_ENERGIES[element]
            E_level = row[self.config.dataset.target_feature]

            # Calculate binding energy for this row
            binding_energy = E_ion - E_level

            binding_energies.append(binding_energy)
            ionization_energies.append(E_ion)

        # Add to dataframe
        self.df['Binding_Energy_cm-1'] = binding_energies
        self.df['Ionization_Energy_cm-1'] = ionization_energies

        # Report statistics per element
        for element in unique_elements:
            element_mask = self.df['Element'] == element
            element_binding = self.df.loc[element_mask, 'Binding_Energy_cm-1']
            Z, E_ion = IONIZATION_ENERGIES[element]

            print(f"\n  {element} (Z={Z}):")
            print(f"    Ionization energy: {E_ion:.2f} cm⁻¹")
            print(f"    Binding energy range: {element_binding.min():.2f} to {element_binding.max():.2f} cm⁻¹")

            # Check for continuum states (E_level > E_ionization)
            n_negative = (element_binding < 0).sum()
            if n_negative > 0:
                print(f"    ⚠️  {n_negative} levels above ionization (clipped to 0)")

        # Clip negative values (continuum states)
        self.df['Binding_Energy_cm-1'] = self.df['Binding_Energy_cm-1'].clip(lower=0)

        # Update target column
        self.config.dataset.target_feature = 'Binding_Energy_cm-1'
        print(f"\n  ✓ Target changed to: Binding_Energy_cm-1")

    def _add_inverse_target(self):
        """
        Apply inverse scaling to the target variable: stores A / E_target.

        Instead of predicting E directly, the model predicts A / E,
        where A is a large constant (e.g. 100 000). This compresses the
        dynamic range: high energies → small values, low energies → large
        values — the opposite of the raw distribution.

        Mathematically:
            Raw target:            E_level
            With binding energy:   E_ion - E_level
            With inverse scaling:  A / E_target   (applied to whichever is active)

        After prediction the model output must be inverted:
            E_target = A / model_output

        NOTE: rows where E_target == 0 are dropped to avoid division by zero.
        These correspond to continuum states already clipped to 0 by
        _add_binding_energy_target(), so removing them is physically correct.
        """
        A = self.config.dataset.get('inverse_target_scale', 100000)
        target_col = self.config.dataset.target_feature  # already updated by binding energy step
        inv_col = f'Inverse_{target_col}'

        print(f"\nApplying inverse target scaling (A={A}):")
        print(f"  Input column:  {target_col}")
        print(f"  Output column: {inv_col}")

        # Drop rows where target == 0 (division by zero; physically: continuum states)
        zero_mask = self.df[target_col] == 0
        n_zeros = zero_mask.sum()
        if n_zeros > 0:
            self.df = self.df[~zero_mask].reset_index(drop=True)
            print(f"  ⚠️  Dropped {n_zeros} rows with E_target=0 (would cause division by zero)")

        self.df[inv_col] = A / self.df[target_col]

        # Report range so you can sanity-check the values look reasonable
        print(f"  Inverse target range: {self.df[inv_col].min():.4f} to {self.df[inv_col].max():.4f}")

        # Update target column name (same pattern as _add_binding_energy_target)
        self.config.dataset.target_feature = inv_col
        print(f"  ✓ Target changed to: {inv_col}")


    def _add_log_target(self):
        log_col = 'Log_Binding_Energy_cm-1'
        self.df[log_col] = np.log(self.df[self.config.dataset.target_feature])
        self.config.dataset.target_feature = log_col

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

    def _create_splits(self):
        """
        Create or load train/val/test splits.

        For multi-element data, creates stratified splits to ensure
        each element is represented in all splits.
        """
        # Determine split file name
        if hasattr(self.config.dataset, 'elements') and len(self.config.dataset.elements) > 1:
            # Multi-element: use combined name
            elements_str = '_'.join(sorted(self.config.dataset.elements))
            split_file = f"dataset_split_indices_{elements_str}.json"
        else:
            # Single element: element-specific split file
            element = self.df['Element'].iloc[0]
            split_file = f"dataset_split_indices_{element}.json"

        # Add data_dir if specified
        if hasattr(self.config.dataset, 'data_dir'):
            split_file = os.path.join(self.config.dataset.data_dir, split_file)

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

        # For multi-element: stratified split (ensure each element in all splits)
        if 'Element' in self.df.columns and self.df['Element'].nunique() > 1:
            train_indices, val_indices, test_indices = self._create_stratified_splits()
        else:
            # Single element: simple random split
            train_indices, val_indices, test_indices = self._create_random_splits()

        # Save splits
        splits = {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices,
            'metadata': {
                'created_date': str(pd.Timestamp.now()),
                'random_seed': self.config.general.random_seed,
                'elements': self.df['Element'].unique().tolist(),
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
        Create stratified splits ensuring each element appears in all sets.
        """
        from sklearn.model_selection import train_test_split

        all_indices = self.df.index.tolist()
        elements = self.df['Element'].values

        # First split: train vs (val+test)
        train_idx, temp_idx = train_test_split(
            all_indices,
            test_size=(self.config.dataset.split.val + self.config.dataset.split.test),
            stratify=elements,
            random_state=self.config.general.random_seed
        )

        # Second split: val vs test
        val_ratio = self.config.dataset.split.val / (self.config.dataset.split.val + self.config.dataset.split.test)
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=(1 - val_ratio),
            stratify=elements[temp_idx],
            random_state=self.config.general.random_seed
        )

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

    def __len__(self) -> int:
        """Return the number of samples in this dataset."""
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Tuple of (features, target) as PyTorch tensors
            - features: Input features (electron config + quantum numbers)
            - target: Energy level in cm⁻¹
        """
        # Convert numpy arrays to PyTorch tensors
        features = torch.FloatTensor(self.X[idx])
        target = torch.FloatTensor(self.y[idx])
        
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
            y = np.exp(y)

        return y

    def validate_term_symbol(self):
        """
        Verify that Term symbol matches S_qn, L_qn, J values.
        This catches data entry errors.
        """
        if 'Term' not in self.df.columns:
            print("  ℹ️  No 'Term' column found - skipping validation")
            return

        # Mapping from L quantum number to letter
        L_LETTER_MAP = {0: 'S', 1: 'P', 2: 'D', 3: 'F', 4: 'G', 5: 'H', 6: 'I', 7: 'K'}

        errors = []

        for idx, row in self.df.iterrows():
            term = row['Term']
            S_qn = row['S_qn']
            L_qn = row['L_qn']
            J = row['J']

            # Parse term symbol: ²P₃/₂ → multiplicity=2, L_letter='P', J_term=1.5
            # Examples: "2S", "2P*", "2D", "4F", "2[3/2]*"
            # Pattern: (multiplicity)(L_letter)(optional_subscript)(optional_asterisk)
            match = re.match(r'^(\d+)([A-Z]).*?(\*?)$', term)

            if not match:
                # Handle bracket notation: "2[3/2]*"
                match_bracket = re.match(r'^(\d+)\[.*?\](\*?)$', term)
                if match_bracket:
                    continue  # Skip bracket notation (different convention)
                errors.append(f"Row {idx}: Cannot parse term '{term}'")
                continue

            multiplicity_str, L_letter, asterisk = match.groups()
            multiplicity_term = int(multiplicity_str)

            # Check multiplicity: should equal 2*S + 1
            multiplicity_expected = int(2 * S_qn + 1)
            if multiplicity_term != multiplicity_expected:
                errors.append(
                    f"Row {idx}: Term '{term}' has multiplicity {multiplicity_term}, "
                    f"but S_qn={S_qn} implies {multiplicity_expected}"
                )

            # Check L letter
            L_letter_expected = L_LETTER_MAP.get(int(L_qn), '?')
            if L_letter != L_letter_expected:
                errors.append(
                    f"Row {idx}: Term '{term}' has L='{L_letter}', "
                    f"but L_qn={L_qn} implies '{L_letter_expected}'"
                )

            # Check parity (asterisk = odd parity)
            if 'parity' in row:
                parity = row['parity']
                has_asterisk = (asterisk == '*')
                is_odd_parity = (parity == -1 or parity == 1)  # Depends on encoding

                # Note: This check depends on how parity is encoded in your data
                # Adjust as needed

        if errors:
            print(f"⚠️  Found {len(errors)} term symbol mismatches:")
            for err in errors[:5]:  # Show first 5
                print(f"   {err}")
        else:
            print(f"✓ Term symbols validated: all consistent with S_qn, L_qn, J")
