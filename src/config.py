"""
Configuration settings for the adaptive loss system
Optimized for CPU training on limited hardware
"""

import torch
import numpy as np

# Device configuration - optimized for your Dell G3 3500
DEVICE = torch.device('cpu')

# CPU optimization settings
NUM_THREADS = 4  # Utilize i7 cores efficiently
torch.set_num_threads(NUM_THREADS)
torch.set_num_interop_threads(2)

# Enable MKL optimizations if available
if torch.backends.mkl.is_available():
    torch.backends.mkl.enabled = True

# Random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Memory optimization settings
BATCH_SIZE = 16  # Reduced for 8GB RAM
MAX_MEMORY_MB = 200  # Memory limit for adaptive components
GRADIENT_ACCUMULATION_STEPS = 2

# RL Agent configuration
RL_CONFIG = {
    'state_dim': 6,
    'action_dim': 3,
    'hidden_dim': 64,
    'learning_rate': 0.001,
    'epsilon': 1.0,
    'epsilon_min': 0.01,
    'epsilon_decay': 0.995,
    'memory_size': 5000,  # Reduced for memory efficiency
    'batch_size': 32
}

# Meta-learning configuration
META_CONFIG = {
    'input_dim': 10,
    'hidden_dim': 64,  # Reduced for efficiency
    'output_dim': 3,
    'learning_rate': 0.001,
    'dropout': 0.2
}

# Training configuration
TRAINING_CONFIG = {
    'num_epochs': 50,
    'batch_size': BATCH_SIZE,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'log_interval': 10,
    'save_interval': 20
}

# Dataset configuration
DATASET_CONFIG = {
    'num_samples': 1000,
    'image_size': 32,
    'num_classes': 5,
    'train_split': 0.8
}

# Visualization settings
VIZ_CONFIG = {
    'plot_interval': 10,
    'save_plots': True,
    'plot_dir': './plots'
}

# Safety mechanisms
SAFETY_CONFIG = {
    'max_loss_multiplier': 5.0,
    'min_loss_multiplier': 0.1,
    'gradient_clip_value': 1.0,
    'stability_threshold': 2.0
}
