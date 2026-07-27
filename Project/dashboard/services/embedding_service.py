"""
Embedding Service Module
------------------------
Extracts 512-dimensional L2-normalized feature embeddings from satellite images
using ModelManager for EuroSAT or UC Merced Fine-Tuned ResNet18 backbones.
"""

import sys
from pathlib import Path
from PIL import Image
import numpy as np

# Ensure project root is in sys.path
SERVICE_DIR = Path(__file__).resolve().parent
DASHBOARD_DIR = SERVICE_DIR.parent
PROJECT_ROOT = DASHBOARD_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from configs.config import DEVICE
from dashboard.services.model_manager import get_embedding_model
from utils.transforms import val_transform


def extract_embedding(image_input, dataset: str = "eurosat") -> np.ndarray:
    """
    Extract a 512-dimensional L2-normalized feature embedding for a given image path or PIL Image.

    Args:
        image_input (str | Path | Image.Image): File path or PIL Image object.
        dataset (str): Model selection ('eurosat' or 'uc_merced').

    Returns:
        np.ndarray: 1D L2-normalized float32 numpy array of shape (512,).
    """
    model = get_embedding_model(dataset)

    if isinstance(image_input, (str, Path)):
        img_path = Path(image_input)
        if not img_path.exists():
            raise FileNotFoundError(f"Image file not found: {img_path}")
        image = Image.open(img_path).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    else:
        raise TypeError(
            f"Unsupported image input type: {type(image_input)}. Expected Path, str, or PIL Image."
        )

    # Preprocess image using exact evaluation transform
    tensor_img = val_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        features = model(tensor_img)  # Shape: (1, 512)

    # Convert to 1D numpy array
    embedding = features.squeeze(0).cpu().numpy().astype(np.float32)

    # Apply L2 Normalization
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding
