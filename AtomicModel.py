"""
AtomicModel.py

This module defines the neural network architecture for predicting atomic energy levels.

Architecture:
- Dense (fully connected) neural network
- Multiple hidden layers with configurable sizes
- Batch normalization for stable training
- Dropout for regularization (prevent overfitting)
- Flexible activation functions (ReLU, LeakyReLU, ELU)

The network takes electron configuration + quantum numbers as input and outputs
a single energy level prediction in cm⁻¹.

Author: Aga
For: Physics project on atomic energy level prediction
"""

import torch
import torch.nn as nn
from typing import List


class DenseAtomicEnergyModel(nn.Module):
    """
    Dense neural network for predicting atomic energy levels.
    
    Architecture:
    Input → [Dense → BatchNorm → Activation → Dropout] × N → Output
    
    Args:
        input_dim: Number of input features (e.g., 40 for electron config + quantum numbers)
        hidden_dims: List of hidden layer sizes (e.g., [128, 64, 32])
        dropout: Dropout probability (0.0 to disable)
        use_batch_norm: Whether to use batch normalization
        activation: Activation function name ('relu', 'leaky_relu', 'elu')
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64, 32],
        dropout: float = 0.3,
        use_batch_norm: bool = True,
        activation: str = 'relu'
    ):
        super(DenseAtomicEnergyModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.use_batch_norm = use_batch_norm
        
        # Select activation function
        # ReLU: most common, fast, works well in most cases
        # LeakyReLU: prevents "dying ReLU" problem (neurons stuck at 0)
        # ELU: smoother gradients, can help with training stability
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1)
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build the network layer by layer
        layers = []
        
        # Input dimension for the first layer
        prev_dim = input_dim
        
        # Create hidden layers
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear (fully connected) layer: transforms input to output dimension
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization: normalizes activations, stabilizes training
            # Helps the network train faster and generalize better
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation function: introduces non-linearity
            # Without this, stacking linear layers would be equivalent to a single layer
            layers.append(self.activation)
            
            # Dropout: randomly zeros some activations during training
            # This prevents the network from relying too much on specific neurons
            # and helps it generalize to new data (reduces overfitting)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Output layer: single neuron for regression (predicts energy level)
        # TODO: final layer output could be passed through torch.clamp(output, min=some_positive_value)
        # No activation function here - we want unbounded output
        layers.append(nn.Linear(prev_dim, 1))
        
        # Combine all layers into a sequential model
        self.network = nn.Sequential(*layers)
        
        # Initialize weights using Xavier/Kaiming initialization
        # This helps the network start training with good gradient flow
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize network weights for better training.
        
        Uses Kaiming initialization for layers with ReLU/LeakyReLU/ELU,
        which accounts for the activation function's behavior.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Kaiming initialization: designed for ReLU-like activations
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                # Batch norm: start with identity transform
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute energy level prediction from input features.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
               Contains electron configuration + quantum numbers
        
        Returns:
            Predicted energy levels of shape (batch_size, 1) in cm⁻¹
        """
        return self.network(x)
    
    def get_num_parameters(self) -> int:
        """
        Count the total number of trainable parameters in the model.
        
        This is useful for understanding model complexity and comparing
        different architectures.
        
        Returns:
            Total number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MultiTaskAtomicModel(nn.Module):
    """
    Multi-task neural network that simultaneously predicts two physical observables:
      1. Atomic energy level (cm⁻¹) — primary task, normalised during training
      2. Landé g-factor gJ            — secondary task, raw dimensionless units

    Architecture — shared trunk + two independent output heads:

        Input
          └─ [Linear → (BatchNorm) → Activation → Dropout] × N   ← shared trunk
                ├─ energy_head: Linear(hidden_dims[-1], 1)          ← energy output
                └─ gj_head:     Linear(hidden_dims[-1], 1)          ← gJ output

    Physics motivation:
        The shared trunk learns a joint representation of the electronic structure
        (orbital occupancies, quantum numbers, coupling products).  This representation
        captures both the energy landscape AND the magnetic moment landscape of the
        atom — two complementary views of the same underlying quantum state.
        Sharing the trunk imposes an inductive bias that similar electron configurations
        should have related energy AND gJ values, which mirrors the physics (both
        observables are derived from the same wave-function via Hund's rules and
        the Landé interval rule).
        The secondary gJ task acts as an auxiliary regulariser: it prevents the energy
        head from overfitting by forcing the trunk to encode physically meaningful
        features beyond what energy alone requires.

    Args: identical to DenseAtomicEnergyModel for drop-in compatibility.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64, 32],
        dropout: float = 0.3,
        use_batch_norm: bool = True,
        activation: str = 'relu'
    ):
        super(MultiTaskAtomicModel, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.use_batch_norm = use_batch_norm

        # Select activation function (same options as DenseAtomicEnergyModel)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1)
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build shared trunk: identical to DenseAtomicEnergyModel EXCEPT the final
        # nn.Linear(hidden_dims[-1], 1) is omitted — that step is replaced by two heads.
        trunk_layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            trunk_layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                trunk_layers.append(nn.BatchNorm1d(hidden_dim))
            trunk_layers.append(self.activation)
            if dropout > 0:
                trunk_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.trunk = nn.Sequential(*trunk_layers)   # shared representation

        # Two independent output heads: each is a single linear projection from the
        # last hidden dimension.  No activation: regression outputs are unbounded.
        self.energy_head = nn.Linear(hidden_dims[-1], 1)  # predicts normalised energy
        self.gj_head     = nn.Linear(hidden_dims[-1], 1)  # predicts raw gJ

        # Kaiming weight initialisation across trunk and both heads
        self._init_weights()

    def _init_weights(self):
        """Kaiming initialisation for all Linear layers; identity init for BatchNorm."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor):
        """
        Forward pass: compute both energy and gJ predictions from input features.

        Args:
            x: shape (batch_size, input_dim)

        Returns:
            (energy_pred, gj_pred) — both shape (batch_size, 1)
            energy_pred: normalised energy (undo with dataset.inverse_transform_target)
            gj_pred:     raw gJ prediction (already in physical units)
        """
        shared = self.trunk(x)              # shared electronic-structure representation
        energy_pred = self.energy_head(shared)   # energy branch
        gj_pred     = self.gj_head(shared)       # gJ branch
        return energy_pred, gj_pred

    def get_num_parameters(self) -> int:
        """Total number of trainable parameters across trunk and both heads."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(config, input_dim: int) -> nn.Module:
    """
    Factory function to create the appropriate model based on config.
    
    Args:
        config: Configuration object
        input_dim: Number of input features
    
    Returns:
        PyTorch model ready for training
    """
    if config.model.architecture == 'dense_nn':
        is_multitask = config.training.get('multitask_gj', False)

        if is_multitask:
            # Multi-task model: shared trunk splits into energy head + gJ head
            model = MultiTaskAtomicModel(
                input_dim=input_dim,
                hidden_dims=config.model.hidden_layers,
                dropout=config.model.dropout,
                use_batch_norm=config.model.use_batch_norm,
                activation=config.model.activation
            )
            alpha = config.training.get('multitask_alpha', 0.9)
            num_params = model.get_num_parameters()
            hidden_str = ' → '.join(map(str, config.model.hidden_layers))
            print(f"\nCreated Multi-Task Dense NN with {num_params:,} trainable parameters")
            print(f"Architecture: {input_dim} → {hidden_str} → [energy_head | gj_head]")
            print(f"Multi-task mode: predicting energy (α={alpha:.2f}) + gJ (α={1 - alpha:.2f})")
        else:
            # Single-task model: identical to original behaviour
            model = DenseAtomicEnergyModel(
                input_dim=input_dim,
                hidden_dims=config.model.hidden_layers,
                dropout=config.model.dropout,
                use_batch_norm=config.model.use_batch_norm,
                activation=config.model.activation
            )
            num_params = model.get_num_parameters()
            print(f"\nCreated Dense NN with {num_params:,} trainable parameters")
            print(f"Architecture: {input_dim} → {' → '.join(map(str, config.model.hidden_layers))} → 1")

        return model
    else:
        raise ValueError(f"Unknown architecture: {config.model.architecture}")
