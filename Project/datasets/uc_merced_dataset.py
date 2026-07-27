"""
UC Merced Dataset Loader & Data Augmentation Module
---------------------------------------------------
Loads UC Merced Land Use images via ImageFolder, applies data augmentations,
and splits into 70% Train, 15% Validation, and 15% Test DataLoaders.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
DATASET_MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DATASET_MODULE_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

from configs.uc_merced_config import (
    BATCH_SIZE,
    DATASET_DIR,
    NUM_WORKERS,
    PIN_MEMORY,
    RANDOM_SEED,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VAL_SPLIT,
)
from utils.transforms import IMAGENET_MEAN, IMAGENET_STD

# Data Augmentations for Training Split
ucm_train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)

# Evaluation Preprocessing for Validation/Test Splits
ucm_val_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


class TransformedSubset(Dataset):
    """
    Wrapper for PyTorch Dataset subsets allowing custom transforms per split.
    """

    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


def get_uc_merced_dataloaders():
    """
    Load UC Merced dataset and create 70/15/15 train/val/test DataLoaders.

    Returns:
        tuple: (train_loader, val_loader, test_loader, class_names)
    """
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"UC Merced dataset directory not found at: {DATASET_DIR}")

    # Load raw PIL images without transformation to apply split-specific transforms later
    raw_dataset = ImageFolder(root=DATASET_DIR)
    class_names = raw_dataset.classes

    total_len = len(raw_dataset)
    train_len = int(total_len * TRAIN_SPLIT)
    val_len = int(total_len * VAL_SPLIT)
    test_len = total_len - train_len - val_len

    # Reproducible random split
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_sub, val_sub, test_sub = random_split(
        raw_dataset, [train_len, val_len, test_len], generator=generator
    )

    # Wrap subsets with respective transformations
    train_ds = TransformedSubset(train_sub, transform=ucm_train_transform)
    val_ds = TransformedSubset(val_sub, transform=ucm_val_transform)
    test_ds = TransformedSubset(test_sub, transform=ucm_val_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    return train_loader, val_loader, test_loader, class_names
