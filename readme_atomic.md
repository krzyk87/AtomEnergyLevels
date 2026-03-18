# Atomic Energy Level Prediction with Neural Networks

A machine learning project for predicting atomic energy levels (in cm⁻¹) from electron configurations and quantum numbers using dense neural networks.

**Author:** Aga (ML Developer)  
**For:** Physics PhD research project at Poznań University of Technology

---

## 📋 Project Overview

This project applies neural networks to predict atomic energy levels, reducing the computational cost of traditional semi-empirical methods. The goal is to:

1. **Predict energy levels** from electron configuration, quantum numbers (J, S, L), and atomic properties
2. **Provide uncertainty estimates** for predictions
3. **Compare** neural network performance with classical ML baselines
4. **Enable rapid prediction** for configurations not yet measured experimentally

### Key Features

- ✅ Dense neural network with configurable architecture
- ✅ Feature engineering for physical properties
- ✅ Data normalization and preprocessing
- ✅ Early stopping and learning rate scheduling
- ✅ Comprehensive evaluation metrics
- ✅ Visualization tools for analysis
- ✅ Well-documented code for physics students

---

## 🗂️ Project Structure

```
atomic_energy_prediction/
├── config_atomic.yaml       # Configuration file
├── main.py                  # Main training/testing script
├── train_model.py          # Training logic
├── test_model.py           # Evaluation logic
├── AtomicDataset.py        # Dataset class
├── AtomicModel.py          # Neural network model
├── utils.py                # Utility functions
├── visualize.py            # Visualization tools
├── energy_Na_features.csv  # Input data (sodium example)
├── saved_models/           # Saved model checkpoints
└── visualizations/         # Output plots
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn pyyaml omegaconf
```

### 2. Prepare Data

Your input CSV should have columns for:
- Electron configuration (e.g., `1s`, `2s`, `2p`, `3s`, `3p`, etc.)
- Quantum numbers (`J`, `S_qn`, `L_qn`, `parity`)
- Atomic properties (`Z`, `A`, `proton_number`, `neutron_number`)
- Target: `Level (cm-1)`

Example: `energy_Na_features.csv` (already provided)

### 3. Train and Test

```bash
# Train and test in one go (recommended for first run)
python main.py

# Train only
python main.py --train_only

# Test only (requires trained model)
python main.py --test_only

# Use custom config
python main.py --config my_config.yaml
```

### 4. Create Visualizations

```bash
python visualize.py
```

---

## ⚙️ Configuration

Edit `config_atomic.yaml` to customize:

### Model Architecture
```yaml
model:
  architecture: dense_nn
  hidden_layers: [128, 64, 32]  # Layer sizes
  dropout: 0.3                  # Dropout rate
  activation: relu              # relu / leaky_relu / elu
  use_batch_norm: true
```

### Training Parameters
```yaml
general:
  epochs: 100
  batch_size: 16
  lr: 0.001                     # Learning rate
  weight_decay: 0.0001          # L2 regularization
  patience: 20                  # Early stopping
```

### Data Processing
```yaml
dataset:
  normalize_features: true      # Standardize inputs
  normalize_target: true        # Standardize outputs
  add_derived_features: true    # Add total_electrons, etc.
```

---

## 📊 Understanding the Output

### Training Output
```
Epoch   1/100 | Train Loss: 0.8523 | Val Loss: 0.7234 | Val MAE: 1250.45 cm⁻¹
Epoch  10/100 | Train Loss: 0.3412 | Val Loss: 0.3198 | Val MAE:  512.33 cm⁻¹
  → New best model! Val Loss: 0.3198
```

- **Train Loss:** How well the model fits training data
- **Val Loss:** How well it generalizes to unseen data
- **Val MAE:** Average prediction error in cm⁻¹
- **New best model:** Saved when validation improves

### Test Results
```
TEST SET PERFORMANCE
Mean Squared Error (MSE):         342156.23
Root Mean Squared Error (RMSE):      584.94 cm⁻¹
Mean Absolute Error (MAE):            412.57 cm⁻¹
R² Score:                               0.8743
Mean Absolute % Error (MAPE):           3.21%
Maximum Error:                       1823.45 cm⁻¹
```

**Interpretation:**
- **RMSE/MAE:** Average prediction error (lower is better)
- **R²:** Proportion of variance explained (0-1, higher is better)
  - 1.0 = perfect predictions
  - 0.0 = no better than predicting the mean
- **MAPE:** Percentage error (useful for comparing across energy scales)

---

## 📈 Visualizations

The `visualize.py` script creates three key plots:

### 1. Predictions vs. True Values
- Scatter plot showing predicted vs. actual energy levels
- Perfect predictions fall on the red diagonal line
- Spread indicates prediction uncertainty

### 2. Error Distribution
- Histogram of prediction errors
- Should be centered around zero (no systematic bias)
- Narrow distribution = better predictions

### 3. Error vs. Energy Level
- Shows if model performs differently at different energy scales
- Helps identify systematic biases

---

## 🔬 For Physics Students: How Does This Work?

### The Problem
Traditional methods (semi-empirical calculations) for computing atomic energy levels:
- Require solving complex Schrödinger equations
- Are computationally expensive (hours to days)
- Need expert physics knowledge

### The Solution
Neural networks learn patterns from existing data:
1. **Input:** Electron configuration + quantum numbers
2. **Processing:** Multiple layers learn complex relationships
3. **Output:** Predicted energy level

### What the Network Learns

The network discovers relationships like:
- **Hund's rules:** How electron spin affects energy
- **Orbital interactions:** How electrons in different orbitals interact
- **Shell effects:** How filling electron shells changes energy
- **Nuclear charge effects:** Impact of protons on energy levels

### Training Process

1. **Forward Pass:** Input → Network → Prediction
2. **Loss Calculation:** How wrong is the prediction?
3. **Backward Pass:** Calculate gradients (how to improve)
4. **Weight Update:** Adjust network to reduce error
5. **Repeat** for many epochs

### Why This Is Useful

- ✅ **Speed:** Predictions in milliseconds vs. hours
- ✅ **Scalability:** Can predict many configurations quickly
- ✅ **Insights:** Feature importance reveals physical patterns
- ✅ **Guidance:** Narrow experimental search ranges

---

## 📝 Next Steps

### Immediate (MVP - Minimum Viable Product)
- [x] Dense neural network implementation
- [x] Data preprocessing and normalization
- [x] Training with early stopping
- [x] Evaluation metrics and visualization
- [ ] Add classical ML baselines (XGBoost, Random Forest)
- [ ] Cross-validation for robust evaluation

### Short-term Enhancements
- [ ] Feature importance analysis
- [ ] Uncertainty quantification (prediction intervals)
- [ ] Multi-element training (Na, K, Rb, etc.)
- [ ] Hyperparameter tuning with grid search

### Long-term Research
- [ ] Physics-informed neural networks (PINNs)
- [ ] Graph neural networks for electron interactions
- [ ] Transfer learning across elements
- [ ] Symbolic regression for interpretable formulas

---

## 🐛 Troubleshooting

### Out of Memory (GPU)
```yaml
# Reduce batch size in config
general:
  batch_size: 8  # or 4
```

### Model Not Learning
- Check learning rate (try 0.0001 or 0.01)
- Verify data normalization is enabled
- Increase model capacity (more/larger layers)
- Check for data leakage in splits

### Poor Test Performance
- Reduce model size (overfitting)
- Increase dropout rate
- Add more data
- Try classical ML baselines first

---

## 📚 References

1. Ruczkowski et al. (2026) - Semi-empirical determination of radiative parameters for Nb II
2. NIST Atomic Spectra Database - https://www.nist.gov/pml/atomic-spectra-database

---

## 📧 Contact

**Aga** - ML Developer  
**Project:** Atomic Energy Level Prediction  
**Institution:** Poznań University of Technology

---

## 📄 License

This project is developed for academic research. Please cite appropriately if used in publications.
