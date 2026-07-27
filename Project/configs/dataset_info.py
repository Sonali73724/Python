"""
Dataset Information Configuration
----------------------------------
Detailed metadata for EuroSAT and UC Merced datasets and fine-tuned models.
All metrics are populated dynamically from evaluation output JSON files.
If output files are unavailable, default safe placeholders (0.00 / N/A) are used.
Refreshed dynamically on every get_dataset_info() call.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EUROSAT_INFO = {
    "key": "eurosat",
    "name": "EuroSAT Sentinel-2",
    "type": "Satellite Imagery",
    "total_images": 0,
    "total_classes": 0,
    "resolution": "64x64 (Resized to 224x224)",
    "image_format": "RGB JPEG",
    "source": "European Space Agency (ESA) Sentinel-2 Satellite",
    "description": "Land Use and Land Cover (LULC) dataset based on Sentinel-2 satellite imagery covering 13 spectral bands across European territories.",
    "split_ratio": "70% Train / 15% Val / 15% Test",
    "model_name": "Fine-Tuned ResNet18 (EuroSAT)",
    "checkpoint": "checkpoints/resnet18_finetuned_best.pth",
    "training_accuracy": 0.00,
    "precision": 0.00,
    "recall": 0.00,
    "f1_score": 0.00,
    "embedding_dim": 512,
    "epochs": 10,
    "optimizer": "Adam (LR=1e-4)",
    "learning_rate": "0.0001",
    # TODO: Load training_time dynamically from training log file when available
    "training_time": "N/A",
    "dataset_dir": PROJECT_ROOT / "datasets" / "eurosat",
}

UC_MERCED_INFO = {
    "key": "uc_merced",
    "name": "UC Merced Land Use",
    "type": "Aerial Imagery",
    "total_images": 0,
    "total_classes": 0,
    "resolution": "256x256 (Resized to 224x224)",
    "image_format": "RGB TIFF",
    "source": "USGS National Map Urban Area Imagery",
    "description": "High-resolution aerial land-use image dataset manually extracted from large national map imagery covering 21 urban land-use classes.",
    "split_ratio": "70% Train / 15% Val / 15% Test",
    "model_name": "ResNet18 Transfer Learning (UC Merced)",
    "checkpoint": "checkpoints/resnet18_ucmerced_best.pth",
    "training_accuracy": 0.00,
    "precision": 0.00,
    "recall": 0.00,
    "f1_score": 0.00,
    "embedding_dim": 512,
    "epochs": 20,
    "optimizer": "Adam (LR=1e-4)",
    "learning_rate": "0.0001",
    # TODO: Load training_time dynamically from training log file when available
    "training_time": "N/A",
    "dataset_dir": PROJECT_ROOT / "datasets" / "uc_merced",
}


def _update_dataset_info_dynamically():
    """Dynamically refresh dataset image counts, class counts, and metric values from output files on demand."""
    # Check EuroSAT checkpoint file existence
    euro_ckpt = PROJECT_ROOT / "checkpoints" / "resnet18_finetuned_best.pth"
    EUROSAT_INFO["checkpoint"] = "checkpoints/resnet18_finetuned_best.pth" if euro_ckpt.exists() else "N/A"

    # Reset EuroSAT metrics default
    EUROSAT_INFO["training_accuracy"] = 0.00
    EUROSAT_INFO["precision"] = 0.00
    EUROSAT_INFO["recall"] = 0.00
    EUROSAT_INFO["f1_score"] = 0.00

    # Scan EuroSAT dataset directory
    eurosat_dir = EUROSAT_INFO["dataset_dir"]
    if eurosat_dir.exists():
        classes = [d for d in eurosat_dir.iterdir() if d.is_dir()]
        if classes:
            EUROSAT_INFO["total_classes"] = len(classes)
            img_count = sum(len(list(d.glob("*.jpg")) + list(d.glob("*.png")) + list(d.glob("*.tif"))) for d in classes)
            EUROSAT_INFO["total_images"] = img_count

    # Load EuroSAT evaluation metrics dynamically from JSON output file
    euro_metrics_file = PROJECT_ROOT / "outputs" / "resnet18_finetuned" / "metrics.json"
    if euro_metrics_file.exists():
        try:
            with open(euro_metrics_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    acc = data.get("accuracy", 0.00)
                    EUROSAT_INFO["training_accuracy"] = round(float(acc * 100 if acc <= 1.0 else acc), 2)
                    prec = data.get("precision", data.get("macro_precision", 0.00))
                    EUROSAT_INFO["precision"] = round(float(prec * 100 if prec <= 1.0 else prec), 2)
                    rec = data.get("recall", data.get("macro_recall", 0.00))
                    EUROSAT_INFO["recall"] = round(float(rec * 100 if rec <= 1.0 else rec), 2)
                    f1 = data.get("f1_score", data.get("macro_f1", 0.00))
                    EUROSAT_INFO["f1_score"] = round(float(f1 * 100 if f1 <= 1.0 else f1), 2)
        except Exception:
            pass

    # Check UC Merced checkpoint file existence
    ucm_ckpt = PROJECT_ROOT / "checkpoints" / "resnet18_ucmerced_best.pth"
    UC_MERCED_INFO["checkpoint"] = "checkpoints/resnet18_ucmerced_best.pth" if ucm_ckpt.exists() else "N/A"

    # Reset UC Merced metrics default
    UC_MERCED_INFO["training_accuracy"] = 0.00
    UC_MERCED_INFO["precision"] = 0.00
    UC_MERCED_INFO["recall"] = 0.00
    UC_MERCED_INFO["f1_score"] = 0.00

    # Scan UC Merced dataset directory
    ucm_dir = UC_MERCED_INFO["dataset_dir"]
    if ucm_dir.exists():
        classes = [d for d in ucm_dir.iterdir() if d.is_dir()]
        if classes:
            UC_MERCED_INFO["total_classes"] = len(classes)
            img_count = sum(len(list(d.glob("*.jpg")) + list(d.glob("*.png")) + list(d.glob("*.tif"))) for d in classes)
            UC_MERCED_INFO["total_images"] = img_count

    # Load UC Merced evaluation metrics dynamically from JSON output file
    ucm_metrics_file = PROJECT_ROOT / "outputs" / "uc_merced" / "metrics.json"
    if ucm_metrics_file.exists():
        try:
            with open(ucm_metrics_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    acc = data.get("accuracy", 0.00)
                    UC_MERCED_INFO["training_accuracy"] = round(float(acc * 100 if acc <= 1.0 else acc), 2)
                    prec = data.get("precision", data.get("macro_precision", 0.00))
                    UC_MERCED_INFO["precision"] = round(float(prec * 100 if prec <= 1.0 else prec), 2)
                    rec = data.get("recall", data.get("macro_recall", 0.00))
                    UC_MERCED_INFO["recall"] = round(float(rec * 100 if rec <= 1.0 else rec), 2)
                    f1 = data.get("f1_score", data.get("macro_f1", 0.00))
                    UC_MERCED_INFO["f1_score"] = round(float(f1 * 100 if f1 <= 1.0 else f1), 2)
        except Exception:
            pass


def get_dataset_info(dataset_key: str = "eurosat") -> dict:
    """
    Retrieve dataset metadata dictionary for 'eurosat' or 'uc_merced'.
    Refreshes metrics dynamically from disk on every invocation.

    Args:
        dataset_key (str): 'eurosat' or 'uc_merced'.

    Returns:
        dict: Metadata information dictionary.
    """
    _update_dataset_info_dynamically()
    key = str(dataset_key).lower().strip()
    if key in ["uc_merced", "ucmerced", "uc-merced"]:
        return UC_MERCED_INFO
    return EUROSAT_INFO
