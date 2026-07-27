"""
Change Detection Service Module
------------------------------
Service layer performing temporal land-use change detection between two satellite images
using ModelManager for EuroSAT or UC Merced models, 512-dim embedding similarity, and heatmaps.
"""

import json
import sys
import time
from pathlib import Path
from PIL import Image

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for Flask
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
SERVICE_DIR = Path(__file__).resolve().parent
DASHBOARD_DIR = SERVICE_DIR.parent
PROJECT_ROOT = DASHBOARD_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F

try:
    from change_detection.heatmap import generate_heatmap
except ImportError:
    def generate_heatmap(before_path, after_path, output_dir):
        pass

from configs.config import DEVICE
from dashboard.services.model_manager import get_model_and_labels, normalize_dataset_key
from dashboard.services.similarity_service import detect_change as similarity_detect_change
from utils.transforms import val_transform

OUTPUTS_CD_DIR = PROJECT_ROOT / "outputs" / "change_detection"


def predict_single_image(image_path: Path, dataset: str = "eurosat") -> tuple[str, float]:
    """
    Run model prediction on a single image using the specified dataset model.

    Args:
        image_path (Path): Path to image file.
        dataset (str): 'eurosat' or 'uc_merced'.

    Returns:
        tuple[str, float]: Predicted class name and confidence percentage.
    """
    model, class_names = get_model_and_labels(dataset)

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as err:
        raise ValueError(f"Unable to read image at {image_path}: {err}")

    tensor_img = val_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor_img)
        probabilities = F.softmax(outputs, dim=1)
        prob, pred_idx = torch.max(probabilities, dim=1)

    idx = pred_idx.item()
    if idx < len(class_names):
        class_name = class_names[idx]
    else:
        class_name = f"Class_{idx}"

    confidence = round(prob.item() * 100, 2)
    return class_name, confidence


def generate_confidence_chart(conf_before: float, conf_after: float) -> Path:
    """Generate vertical bar chart comparing confidence values."""
    OUTPUTS_CD_DIR.mkdir(parents=True, exist_ok=True)
    chart_path = OUTPUTS_CD_DIR / "confidence_chart.png"

    categories = ["Before Image", "After Image"]
    confidences = [conf_before, conf_after]
    colors = ["#4C72B0", "#55A868"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(categories, confidences, color=colors, width=0.45)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_ylabel("Confidence (%)", fontsize=11)
    ax.set_title("Prediction Confidence Comparison", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(chart_path, dpi=200)
    plt.close()

    return chart_path


from configs.dataset_info import get_dataset_info

def detect_change(
    before_image_path: Path,
    after_image_path: Path,
    threshold: float = None,
    dataset: str = "eurosat",
) -> dict:
    """
    Perform temporal change detection using selected model dataset ('eurosat' or 'uc_merced').
    """
    norm_key = normalize_dataset_key(dataset)
    dataset_display = "UC Merced" if norm_key == "uc_merced" else "EuroSAT"
    model_info = get_dataset_info(norm_key)
    model_name = model_info.get("model_name") or "N/A"

    before_class, conf_before = predict_single_image(
        before_image_path, dataset=norm_key
    )
    after_class, conf_after = predict_single_image(
        after_image_path, dataset=norm_key
    )

    sim_res = similarity_detect_change(
        before_image_path,
        after_image_path,
        threshold=threshold,
        dataset=norm_key,
    )

    similarity = sim_res["similarity"]
    threshold_val = sim_res["threshold"]
    changed = sim_res["change"]
    status = sim_res["message"]

    generate_confidence_chart(conf_before, conf_after)
    generate_heatmap(before_image_path, after_image_path, output_dir=OUTPUTS_CD_DIR)

    timestamp = int(time.time() * 1000)
    result_data = {
        "dataset_used": dataset_display,
        "model_name": model_name,
        "before_class": before_class,
        "after_class": after_class,
        "confidence_before": conf_before,
        "confidence_after": conf_after,
        "embedding_dim": model_info.get("embedding_dim", 512),
        "similarity": similarity,
        "threshold": threshold_val,
        "changed": changed,
        "status": status,
        "chart_url": f"/outputs/change_detection/confidence_chart.png?t={timestamp}",
        "diff_url": f"/outputs/change_detection/difference.png?t={timestamp}",
        "overlay_url": f"/outputs/change_detection/heatmap_overlay.png?t={timestamp}",
    }


    OUTPUTS_CD_DIR.mkdir(parents=True, exist_ok=True)

    json_path = OUTPUTS_CD_DIR / "result.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_used": dataset_display,
                "before_class": before_class,
                "after_class": after_class,
                "confidence_before": conf_before,
                "confidence_after": conf_after,
                "embedding_dim": 512,
                "similarity": similarity,
                "threshold": threshold_val,
                "changed": changed,
                "status": status,
            },
            f,
            indent=4,
        )

    return result_data
