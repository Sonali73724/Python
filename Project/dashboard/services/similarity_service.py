"""
Similarity Service Module
-------------------------
Service layer performing cosine similarity comparison between image embeddings
to make temporal change detection decisions based on optimal thresholds.
Supports EuroSAT and UC Merced embedding models via ModelManager.
"""

import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
SERVICE_DIR = Path(__file__).resolve().parent
DASHBOARD_DIR = SERVICE_DIR.parent
PROJECT_ROOT = DASHBOARD_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from dashboard.services.embedding_service import extract_embedding

THRESHOLD_JSON = PROJECT_ROOT / "outputs" / "change_detection" / "threshold.json"


def get_default_threshold() -> float:
    """
    Load optimal threshold from outputs/change_detection/threshold.json if available.
    Falls back to 0.85 if missing.
    """
    if THRESHOLD_JSON.exists():
        try:
            with open(THRESHOLD_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "best_threshold" in data:
                    return float(data["best_threshold"])
        except Exception as err:
            print(f"[Warning] Could not load threshold.json: {err}")
    return 0.85


def compute_cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Compute cosine similarity between two 1D feature embedding vectors.

    Args:
        embedding1 (np.ndarray): 1D L2-normalized float32 numpy array.
        embedding2 (np.ndarray): 1D L2-normalized float32 numpy array.

    Returns:
        float: Cosine similarity score between 0.0 and 1.0 rounded to 4 decimal places.
    """
    emb1_2d = np.atleast_2d(embedding1)
    emb2_2d = np.atleast_2d(embedding2)

    sim_matrix = cosine_similarity(emb1_2d, emb2_2d)
    sim_score = float(sim_matrix[0, 0])

    sim_score = max(0.0, min(1.0, sim_score))
    return round(sim_score, 4)


def detect_change(
    image1, image2, threshold: float = None, dataset: str = "eurosat"
) -> dict:
    """
    Perform embedding-based temporal change detection between two images using cosine similarity.

    Args:
        image1 (str | Path | Image.Image): First image (file path or PIL Image).
        image2 (str | Path | Image.Image): Second image (file path or PIL Image).
        threshold (float, optional): Similarity decision boundary threshold.
        dataset (str): Model selection ('eurosat' or 'uc_merced').

    Returns:
        dict: Result containing similarity score, threshold, change boolean, and message.
    """
    if threshold is None:
        threshold = get_default_threshold()

    emb1 = extract_embedding(image1, dataset=dataset)
    emb2 = extract_embedding(image2, dataset=dataset)

    similarity = compute_cosine_similarity(emb1, emb2)

    if similarity >= threshold:
        change = False
        message = "No Change"
    else:
        change = True
        message = "Change Detected"

    return {
        "similarity": similarity,
        "threshold": threshold,
        "change": change,
        "message": message,
    }
