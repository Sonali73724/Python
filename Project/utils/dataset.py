from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, Subset

from configs.config import (
    DATASET_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    RANDOM_SEED,
)

from utils.transforms import (
    train_transform,
    val_transform,
)

class SubsetDataset(Dataset):
    """
    Applies transforms independently to a subset.
    """

    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.classes = dataset.classes

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        image, label = self.dataset[self.indices[idx]]

        if self.transform:
            image = self.transform(image)

        return image, label



class EuroSATDataset(ImageFolder):
    """
    Custom EuroSAT Dataset
    Allows different transforms for train and validation.
    """

    def __init__(self, root, transform=None):
        super().__init__(root=root, transform=transform)


def get_dataloaders():

    # -------------------------
    # Load Full Dataset
    # -------------------------

    full_dataset = EuroSATDataset(
        root=DATASET_DIR,
        transform=None
    )

    # -------------------------
    # Dataset Split
    # -------------------------

    total_size = len(full_dataset)

    train_size = int(0.70 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(RANDOM_SEED)

    indices = torch.randperm(total_size, generator=generator).tolist()

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    train_dataset = SubsetDataset(
        full_dataset,
        train_indices,
        transform=train_transform
    )

    val_dataset = SubsetDataset(
        full_dataset,
        val_indices,
        transform=val_transform
    )

    test_dataset = SubsetDataset(
        full_dataset,
        test_indices,
        transform=val_transform
    )


    # -------------------------
    # DataLoaders
    # -------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        full_dataset.classes,
    )