"""
Model Manager Service Module
----------------------------
Centralized model manager loading, caching, and serving EuroSAT and UC Merced PyTorch models.
Ensures models and weights are loaded into memory only once.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
SERVICE_DIR = Path(__file__).resolve().parent
DASHBOARD_DIR = SERVICE_DIR.parent
PROJECT_ROOT = DASHBOARD_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

from configs.class_labels import get_class_labels
from configs.config import DEVICE, CHECKPOINT_DIR
from models.resnet18_model import ResNet18Model
from models.resnet18_ucmerced_model import ResNet18UCMercedModel

CHECKPOINTS = {
    "eurosat": CHECKPOINT_DIR / "resnet18_finetuned_best.pth",
    "uc_merced": CHECKPOINT_DIR / "resnet18_ucmerced_best.pth",
}

# In-memory model caches
_classifier_cache: dict[str, nn.Module] = {}
_embedding_cache: dict[str, nn.Module] = {}


def normalize_dataset_key(dataset_name: str) -> str:
    """Normalize input string to 'eurosat' or 'uc_merced'."""
    key = str(dataset_name).lower().strip()
    if key in ["uc_merced", "ucmerced", "uc-merced", "uc_merced_model"]:
        return "uc_merced"
    return "eurosat"


def get_model(dataset_name: str = "eurosat") -> nn.Module:
    """
    Retrieve cached classifier model for the specified dataset.
    Loads checkpoint weights into memory only on first invocation.

    Args:
        dataset_name (str): 'eurosat' or 'uc_merced'.

    Returns:
        nn.Module: Loaded PyTorch classifier model in eval mode.
    """
    key = normalize_dataset_key(dataset_name)

    if key not in _classifier_cache:
        ckpt_path = CHECKPOINTS[key]

        if key == "uc_merced":
            if not ckpt_path.exists():
                # Fallback to EuroSAT if UC Merced checkpoint missing
                ckpt_path = CHECKPOINTS["eurosat"]
                model = ResNet18Model(num_classes=10).to(DEVICE)
            else:
                model = ResNet18UCMercedModel(num_classes=21).to(DEVICE)
        else:
            if not ckpt_path.exists():
                raise FileNotFoundError(f"EuroSAT checkpoint missing at: {ckpt_path}")
            model = ResNet18Model(num_classes=10).to(DEVICE)

        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        model.eval()
        _classifier_cache[key] = model
        print(f"✅ [ModelManager] Cached classifier model '{key}' from {ckpt_path.name}")

    return _classifier_cache[key]


def get_model_and_labels(dataset_name: str = "eurosat") -> tuple[nn.Module, list[str]]:
    """
    Retrieve classifier model and its corresponding class labels.

    Args:
        dataset_name (str): 'eurosat' or 'uc_merced'.

    Returns:
        tuple[nn.Module, list[str]]: Loaded model and class label list.
    """
    key = normalize_dataset_key(dataset_name)
    model = get_model(key)
    labels = get_class_labels(key)
    return model, labels


def get_embedding_model(dataset_name: str = "eurosat") -> nn.Module:
    """
    Retrieve cached feature embedding model with the final FC layer bypassed.

    Args:
        dataset_name (str): 'eurosat' or 'uc_merced'.

    Returns:
        nn.Module: Feature extractor model in eval mode (outputs 512-dim features).
    """
    key = normalize_dataset_key(dataset_name)

    if key not in _embedding_cache:
        ckpt_path = CHECKPOINTS[key]

        if key == "uc_merced":
            if not ckpt_path.exists():
                ckpt_path = CHECKPOINTS["eurosat"]
                model = ResNet18Model(num_classes=10).to(DEVICE)
            else:
                model = ResNet18UCMercedModel(num_classes=21).to(DEVICE)
        else:
            model = ResNet18Model(num_classes=10).to(DEVICE)

        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        model.model.fc = nn.Identity()
        model.eval()
        _embedding_cache[key] = model
        print(f"✅ [ModelManager] Cached embedding extractor '{key}' from {ckpt_path.name}")

    return _embedding_cache[key]
