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

Author: Aga (ML Developer)
For: Physics PhD project on atomic energy level prediction
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
