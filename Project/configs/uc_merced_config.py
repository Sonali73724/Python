"""
UC Merced Configuration File
----------------------------
Project paths and hyperparameter settings for UC Merced training and evaluation.
"""

from pathlib import Path
import torch

# ======================================================
# Project Paths
# ======================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "datasets" / "uc_merced"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "uc_merced"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "resnet18_ucmerced_best.pth"

# ======================================================
# Dataset Parameters
# ======================================================
IMAGE_SIZE = 224
NUM_CLASSES = 21

TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

RANDOM_SEED = 42

# ======================================================
# Training Hyperparameters
# ======================================================
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
EARLY_STOPPING_PATIENCE = 5

NUM_WORKERS = 0
PIN_MEMORY = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
