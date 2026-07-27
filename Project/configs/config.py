from pathlib import Path
import torch

# ======================================================
# Project Paths
# ======================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "datasets" / "eurosat"

OUTPUT_DIR = PROJECT_ROOT / "outputs"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# ======================================================
# Dataset
# ======================================================

IMAGE_SIZE = 224
NUM_CLASSES = 10

# ======================================================
# DataLoader
# ======================================================

BATCH_SIZE = 32
NUM_WORKERS = 4
PIN_MEMORY = True

# ======================================================
# Training
# ======================================================

RANDOM_SEED = 42

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# ======================================================
# Baseline CNN
# ======================================================

BASELINE_EPOCHS = 5
RESNET_EPOCHS = 5
# ======================================================
# Transfer Learning
# ======================================================

FREEZE_EPOCHS = 5
FINETUNE_EPOCHS = 5

LR_PHASE1 = 1e-3
LR_PHASE2 = 1e-4
NUM_WORKERS = 4
PIN_MEMORY = True