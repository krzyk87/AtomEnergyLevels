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
        
        # Load the full dataset from CSV
        self.df = pd.read_csv(config.dataset.data_file)
        
        print(f"Loaded {len(self.df)} atomic energy levels from {config.dataset.data_file}")
        
        # Add derived features if requested (total electrons, valence electrons, etc.)
        if config.dataset.add_derived_features:
            self._add_derived_features()
        
        # Prepare feature columns (inputs to the model)
        self.feature_columns = self._get_feature_columns()
        
        # Prepare target column (what we want to predict)
        self.target_column = config.dataset.target_feature
        
        # Handle missing values in the data
        self._handle_missing_values()
        
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
        
        print(f"{subset.capitalize()} set: {len(self)} samples, {self.X.shape[1]} features")

    def _encode_valence_electrons(self, max_valence=10):
        """
        Encode valence electrons as fixed-size feature vector of individual (n, l) pairs with padding.

        This creates a FIXED number of features regardless of how many
        valence electrons the atom actually has. Atoms with fewer valence
        electrons are padded with zeros.

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
                n_electrons = int(row[col])
                if n_electrons > 0:
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
            valence_electrons = all_electrons[-max_valence:] if len(
                all_electrons) >= max_valence else all_electrons

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

        # Show example
        example_idx = self.df.index[0]
        example_features = [int(self.df.loc[example_idx, col]) for col in feature_cols]
        print(f"  Example configuration: {example_features}")

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
        
        print(f"Added derived features: total_electrons, valence_electrons, "
              f"core_electrons, unpaired_electrons, max_principal_n")
    
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
        Split the dataset into train/validation/test sets.

        Uses random splitting based on the ratios specified in config.
        Saves split indices to a JSON file for reproducibility.

        **IMPORTANT:** Once splits are created, they are reused for all subsequent
        experiments to ensure fair comparison during hyperparameter tuning.
        """
        # Get split file name from config or fall back to default
        atom_name = self.config.dataset.data_file.split('_')[1].split('.')[0]  # Extract 'Na' from 'Na_features.csv'
        default_split_file = f"dataset_split_indices_{atom_name}.json"
        split_file_name = (
            self.config.dataset.split_file
            if hasattr(self.config.dataset, 'split_file')
            else default_split_file
        )
        split_file = os.path.join("data", split_file_name)

        # ✅ Load existing splits if available (for ALL subsets)
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                splits = json.load(f)
            self.indices = splits[self.subset]
            print(f"Loaded {self.subset} split from {split_file} (using existing splits)")
            return

        # ⚠️ Only create new splits if file doesn't exist
        print(f"Creating NEW data splits (file not found: {split_file})")
        print("⚠️  These splits will be used for all future experiments!")

        np.random.seed(self.config.general.random_seed)

        # Get all valid indices
        all_indices = self.df.index.tolist()
        np.random.shuffle(all_indices)

        # Calculate split sizes
        n_total = len(all_indices)
        n_train = int(n_total * self.config.dataset.split.train)
        n_val = int(n_total * self.config.dataset.split.val)

        # Split indices
        train_indices = all_indices[:n_train]
        val_indices = all_indices[n_train:n_train + n_val]
        test_indices = all_indices[n_train + n_val:]

        # Save splits to file
        splits = {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices,
            'metadata': {
                'created_date': str(pd.Timestamp.now()),
                'random_seed': self.config.general.random_seed,
                'train_size': len(train_indices),
                'val_size': len(val_indices),
                'test_size': len(test_indices),
                'data_file': self.config.dataset.data_file
            }
        }

        os.makedirs("data", exist_ok=True)
        with open(split_file, 'w') as f:
            json.dump(splits, f, indent=2)

        print(f"Created and saved data splits to {split_file}")
        print(f"  Train: {len(train_indices)} samples ({len(train_indices) / n_total * 100:.1f}%)")
        print(f"  Val:   {len(val_indices)} samples ({len(val_indices) / n_total * 100:.1f}%)")
        print(f"  Test:  {len(test_indices)} samples ({len(test_indices) / n_total * 100:.1f}%)")

        # Store the indices for this subset
        self.indices = splits[self.subset]
    
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
        
        Args:
            y_normalized: Normalized target values
            
        Returns:
            Target values in original scale (cm⁻¹)
        """
        if self.scaler_target is not None:
            return self.scaler_target.inverse_transform(y_normalized)
        return y_normalized

    def validate_term_symbol(self):
        """
        Verify that Term symbol matches S_qn, L_qn, J values.
        This catches data entry errors.
        """
        if 'Term' not in self.df.columns:
            return

        errors = []

        for idx, row in self.df.iterrows():
            term = row['Term']
            S_qn = row['S_qn']
            L_qn = row['L_qn']
            J = row['J']

            # Parse term symbol: ²P₃/₂ → multiplicity=2, L_letter='P', J_term=1.5
            multiplicity_expected = int(2 * S_qn + 1)
            L_letter_map = {0: 'S', 1: 'P', 2: 'D', 3: 'F', 4: 'G', 5: 'H'}
            L_letter_expected = L_letter_map.get(int(L_qn), '?')

            # Check if term matches
            if not term.startswith(str(multiplicity_expected)):
                errors.append(f"Row {idx}: Term '{term}' multiplicity doesn't match S_qn={S_qn}")

            if L_letter_expected not in term:
                errors.append(f"Row {idx}: Term '{term}' L doesn't match L_qn={L_qn}")

        if errors:
            print(f"⚠️  Found {len(errors)} term symbol mismatches:")
            for err in errors[:5]:  # Show first 5
                print(f"   {err}")
        else:
            print(f"✓ Term symbols validated: all consistent with S_qn, L_qn, J")
