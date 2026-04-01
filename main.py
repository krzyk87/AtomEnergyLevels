"""
main.py

Main entry point for training and testing atomic energy prediction models.

Usage:
    # Train and test in one go
    python main.py
    
    # Train only
    python main.py --train_only
    
    # Test only (requires trained model)
    python main.py --test_only
    
    # Use custom config
    python main.py --config my_config.yaml

Author: Aga
For: Physics project on atomic energy level prediction
"""

import argparse
import os
import sys

from utils import load_config, set_seed, check_cuda, get_model_name_from_config
from train_model import train_one_run
from test_model import test_one_run


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and test atomic energy prediction models"
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config_atomic.yaml',
        help='Path to configuration file (default: config_atomic.yaml)'
    )
    
    parser.add_argument(
        '--train_only',
        action='store_true',
        help='Only run training (skip testing)'
    )
    
    parser.add_argument(
        '--test_only',
        action='store_true',
        help='Only run testing (skip training)'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint for testing'
    )
    
    return parser.parse_args()


def main():
    """
    Main function to run training and/or testing.
    
    Workflow:
    1. Load configuration
    2. Set random seeds for reproducibility
    3. Check GPU availability
    4. Train model (if not --test_only)
    5. Test model (if not --train_only)
    """
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    print("\n" + "="*60)
    print("ATOMIC ENERGY LEVEL PREDICTION")
    print("="*60 + "\n")
    
    config = load_config(args.config)
    
    # Set random seed for reproducibility
    # This ensures experiments can be exactly reproduced
    set_seed(config.general.random_seed)
    
    # Check if GPU is available
    check_cuda()
    
    print("\n" + "-"*60)
    print("CONFIGURATION")
    print("-"*60)
    # print(f"Data file: {config.dataset.data_file}")
    print(f"Model: {config.model.architecture}")
    print(f"Hidden layers: {config.model.hidden_layers}")
    print(f"Batch size: {config.general.batch_size}")
    print(f"Learning rate: {config.general.lr}")
    print(f"Epochs: {config.general.epochs}")
    print(f"Device: {config.general.device}")
    print("-"*60 + "\n")
    
    best_model_path = None
    
    # Training phase
    if not args.test_only:
        print("="*60)
        print("TRAINING PHASE")
        print("="*60 + "\n")
        
        try:
            best_model_path = train_one_run(config)
            print("\n✓ Training completed successfully")
        except Exception as e:
            print(f"\n✗ Training failed with error: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    # Testing phase
    if not args.train_only:
        print("\n" + "="*60)
        print("TESTING PHASE")
        print("="*60 + "\n")
        
        # Use specified checkpoint or the one from training
        checkpoint = args.checkpoint if args.checkpoint else best_model_path
        
        if checkpoint is None:
            checkpoint = os.path.join(config.logging.save_dir, get_model_name_from_config(config))
        
        if not os.path.exists(checkpoint):
            print(f"✗ Checkpoint not found: {checkpoint}")
            print("  Please train a model first or specify --checkpoint")
            return 1
        
        try:
            metrics = test_one_run(config, checkpoint)
            print("✓ Testing completed successfully")
            
            # Print summary
            print("\n" + "="*60)
            print("SUMMARY")
            print("="*60)
            print(f"Best model: {checkpoint}")
            print(f"Test RMSE: {metrics['rmse']:.2f} cm⁻¹")
            print(f"Test MAE:  {metrics['mae']:.2f} cm⁻¹")
            print(f"Test R²:   {metrics['r2']:.4f}")
            print("="*60 + "\n")
            
        except Exception as e:
            print(f"\n✗ Testing failed with error: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    print("All tasks completed successfully! 🎉\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
