"""Configuration and global settings for Diffusion Models project."""

import torch

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
DATA_ROOT = "data/raw"
MODEL_SAVE_PATH = "models/checkpoints"

# Dataset
DATASET_NAME = "MNIST"
IMAGE_SIZE = 28
IMAGE_CHANNELS = 1
BATCH_SIZE = 128
N_EPOCHS = 3
LEARNING_RATE = 1e-3

# U-Net Model
UNET_CONFIG = {
    "sample_size": 28,
    "in_channels": 1,
    "out_channels": 1,
    "layers_per_block": 2,
    "block_out_channels": (32, 64, 64),
    "down_block_types": (
        "DownBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D",
    ),
    "up_block_types": (
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
    ),
}

# DDPM Scheduler
NUM_TRAIN_TIMESTEPS = 1000

# Sampling
SAMPLING_STEPS = 40
NUM_SAMPLES = 64