"""
Metrics Service Module
---------------------
Service layer responsible for loading dynamic EuroSAT and UC Merced dataset metadata,
training curves, evaluation metrics, model comparison metrics, class distribution counts,
sample images, and prediction progression comparisons.
"""

import json
import sys
from pathlib import Path
from PIL import Image

# Base paths: resolve project root relative to dashboard/services/metrics_service.py
SERVICE_DIR = Path(__file__).resolve().parent
DASHBOARD_DIR = SERVICE_DIR.parent
PROJECT_ROOT = DASHBOARD_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F

from configs.class_labels import EUROSAT_CLASSES, get_class_labels
from configs.config import CHECKPOINT_DIR, DEVICE
from configs.dataset_info import EUROSAT_INFO, UC_MERCED_INFO, get_dataset_info
from utils.transforms import val_transform

OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def get_sample_images_for_dataset(dataset_key: str) -> list[dict]:
    """Dynamically scan dataset folder and return one sample image per class."""
    info = get_dataset_info(dataset_key)
    dataset_dir = info["dataset_dir"]
    samples = []

    if dataset_dir.exists():
        for class_dir in sorted(dataset_dir.iterdir()):
            if class_dir.is_dir():
                imgs = sorted(
                    list(class_dir.glob("*.jpg"))
                    + list(class_dir.glob("*.png"))
                    + list(class_dir.glob("*.tif"))
                )
                if len(imgs) > 0:
                    sample_img = imgs[0]
                    rel_path = sample_img.relative_to(PROJECT_ROOT / "datasets")
                    samples.append({
                        "class_name": class_dir.name,
                        "file_name": sample_img.name,
                        "image_url": f"/datasets/{rel_path}",
                    })

    return samples


def get_class_distribution_for_dataset(dataset_key: str) -> dict:
    """Scan dataset directory and count total images per class directory."""
    info = get_dataset_info(dataset_key)
    dataset_dir = info["dataset_dir"]
    class_counts = {}

    if dataset_dir.exists():
        for class_dir in sorted(dataset_dir.iterdir()):
            if class_dir.is_dir():
                imgs = (
                    list(class_dir.glob("*.jpg"))
                    + list(class_dir.glob("*.png"))
                    + list(class_dir.glob("*.tif"))
                )
                class_counts[class_dir.name] = len(imgs)

    return {
        "classes": list(class_counts.keys()),
        "counts": list(class_counts.values()),
        "total_images": sum(class_counts.values()),
    }


def get_dataset_page_data(dataset_key: str = "eurosat") -> dict:
    """Consolidate all data for /dataset view."""
    info = get_dataset_info(dataset_key)
    class_labels = get_class_labels(dataset_key)
    samples = get_sample_images_for_dataset(dataset_key)
    distribution = get_class_distribution_for_dataset(dataset_key)

    holdout_info = {
        "name": "UC Merced Land Use",
        "purpose": "Independent Generalization Evaluation",
        "total_images": 2100,
        "total_classes": 21,
    }

    return {
        "info": info,
        "class_labels": class_labels,
        "sample_images": samples,
        "distribution": distribution,
        "holdout_info": holdout_info,
    }



def get_training_page_data(dataset_key: str) -> dict:
    """Consolidate all data for /training view."""
    info = get_dataset_info(dataset_key)
    key = info["key"]

    if key == "uc_merced":
        loss_curve = "/outputs/uc_merced/loss_curve.png"
        acc_curve = "/outputs/uc_merced/accuracy_curve.png"
    else:
        loss_curve = "/outputs/resnet18_finetuned/loss_curve.png"
        acc_curve = "/outputs/resnet18_finetuned/accuracy_curve.png"

    return {
        "info": info,
        "loss_curve_url": loss_curve,
        "accuracy_curve_url": acc_curve,
    }


def get_evaluation_page_data(dataset_key: str) -> dict:
    """Consolidate evaluation metrics for /evaluation view."""
    info = get_dataset_info(dataset_key)
    key = info["key"]

    if key == "uc_merced":
        out_dir = OUTPUTS_DIR / "uc_merced"
        matrix_url = "/outputs/uc_merced/confusion_matrix.png"
    else:
        out_dir = OUTPUTS_DIR / "resnet18_finetuned"
        matrix_url = "/outputs/resnet18_finetuned/confusion_matrix.png"

    metrics_file = out_dir / "metrics.json"
    report_file = out_dir / "classification_report.txt"

    metrics = {
        "accuracy": 0.00,
        "precision": 0.00,
        "recall": 0.00,
        "f1_score": 0.00,
    }

    if metrics_file.exists():
        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    acc = data.get("accuracy", 0.00)
                    metrics["accuracy"] = round(float(acc * 100 if acc <= 1.0 else acc), 2)
                    prec = data.get("precision", data.get("macro_precision", 0.00))
                    metrics["precision"] = round(float(prec * 100 if prec <= 1.0 else prec), 2)
                    rec = data.get("recall", data.get("macro_recall", 0.00))
                    metrics["recall"] = round(float(rec * 100 if rec <= 1.0 else rec), 2)
                    f1 = data.get("f1_score", data.get("macro_f1", 0.00))
                    metrics["f1_score"] = round(float(f1 * 100 if f1 <= 1.0 else f1), 2)
        except Exception:
            pass


    report_text = ""
    if report_file.exists():
        try:
            with open(report_file, "r", encoding="utf-8") as f:
                report_text = f.read()
        except Exception:
            pass

    return {
        "info": info,
        "metrics": metrics,
        "report_text": report_text,
        "matrix_url": matrix_url,
    }


def get_merged_evaluation_data() -> dict:
    """
    Consolidate both EuroSAT Validation and UC Merced Holdout Evaluation metrics
    into a single structured response for unified evaluation page rendering.
    """
    eurosat_eval = get_evaluation_page_data("eurosat")
    
    from services.uc_merced_service import get_uc_merced_metrics
    ucm_raw = get_uc_merced_metrics()
    
    ucm_eval = {
        "info": get_dataset_info("uc_merced"),
        "metrics": {
            "accuracy": ucm_raw.get("accuracy_pct", ucm_raw.get("accuracy", 0.00)),
            "precision": ucm_raw.get("precision", 0.00),
            "recall": ucm_raw.get("recall", 0.00),
            "f1_score": ucm_raw.get("f1_score", 0.00),
        },
        "report_text": ucm_raw.get("classification_report", ""),
        "matrix_url": ucm_raw.get("matrix_url", None),
    }


    return {
        "eurosat": eurosat_eval,
        "uc_merced": ucm_eval,
    }



def get_model_prediction(model_class, checkpoint_path: Path, image_path: Path) -> tuple[str, float]:
    """Helper to load a checkpoint and predict class & confidence for a single test image."""
    try:
        model = model_class().to(DEVICE)
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        model.eval()

        img = Image.open(image_path).convert("RGB")
        tensor_img = val_transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(tensor_img)
            probs = F.softmax(outputs, dim=1)
            prob, pred_idx = torch.max(probs, dim=1)

        idx = pred_idx.item()
        cls_name = EUROSAT_CLASSES[idx] if idx < len(EUROSAT_CLASSES) else f"Class_{idx}"
        confidence = round(prob.item() * 100, 2)
        return cls_name, confidence
    except Exception as err:
        print(f"[Warning] Prediction failed for {checkpoint_path.name}: {err}")
        return "Unknown", 0.0


def get_comparison_page_data() -> dict:
    """
    Consolidate 3-Model EuroSAT Performance Comparison Data:
    1. Baseline CNN
    2. ResNet18 (Frozen)
    3. ResNet18 (Fine-Tuned)
    Metrics are loaded dynamically from outputs/*/metrics.json. Fallback is strictly 0.00 / N/A.
    """
    # Initialize model evaluation metrics with 0.00 (loaded dynamically from outputs/ if present)
    m_baseline = {"accuracy": 0.00, "precision": 0.00, "recall": 0.00, "f1_score": 0.00}
    m_frozen = {"accuracy": 0.00, "precision": 0.00, "recall": 0.00, "f1_score": 0.00}
    m_finetuned = {"accuracy": 0.00, "precision": 0.00, "recall": 0.00, "f1_score": 0.00}

    p_baseline = OUTPUTS_DIR / "baseline_cnn" / "metrics.json"
    p_frozen = OUTPUTS_DIR / "resnet18_frozen" / "metrics.json"
    p_finetuned = OUTPUTS_DIR / "resnet18_finetuned" / "metrics.json"

    for path, target in [(p_baseline, m_baseline), (p_frozen, m_frozen), (p_finetuned, m_finetuned)]:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        acc = data.get("accuracy", data.get("best_val_acc", data.get("final_val_acc", 0.00)))
                        target["accuracy"] = round(float(acc * 100 if acc <= 1.0 else acc), 2)
                        prec = data.get("precision", data.get("macro_precision", 0.00))
                        target["precision"] = round(float(prec * 100 if prec <= 1.0 else prec), 2)
                        rec = data.get("recall", data.get("macro_recall", 0.00))
                        target["recall"] = round(float(rec * 100 if rec <= 1.0 else rec), 2)
                        f1 = data.get("f1_score", data.get("macro_f1", 0.00))
                        target["f1_score"] = round(float(f1 * 100 if f1 <= 1.0 else f1), 2)
            except Exception:
                pass

    imp_baseline = round(m_finetuned["accuracy"] - m_baseline["accuracy"], 2)
    imp_frozen = round(m_finetuned["accuracy"] - m_frozen["accuracy"], 2)

    ckpt_base = CHECKPOINT_DIR / "baseline_cnn_best.pth"
    ckpt_froz = CHECKPOINT_DIR / "resnet18_frozen_best.pth"
    ckpt_fine = CHECKPOINT_DIR / "resnet18_finetuned_best.pth"

    ckpt_base_size = f"{ckpt_base.stat().st_size / (1024 * 1024):.1f} MB" if ckpt_base.exists() else "N/A"
    ckpt_froz_size = f"{ckpt_froz.stat().st_size / (1024 * 1024):.1f} MB" if ckpt_froz.exists() else "N/A"
    ckpt_fine_size = f"{ckpt_fine.stat().st_size / (1024 * 1024):.1f} MB" if ckpt_fine.exists() else "N/A"

    # TODO: Load training duration, inference speed, and parameters dynamically from training logs when available
    metrics_table = [
        {"metric": "Overall Accuracy", "baseline": f"{m_baseline['accuracy']:.2f}%", "frozen": f"{m_frozen['accuracy']:.2f}%", "finetuned": f"{m_finetuned['accuracy']:.2f}%"},
        {"metric": "Macro Precision", "baseline": f"{m_baseline['precision']:.2f}%", "frozen": f"{m_frozen['precision']:.2f}%", "finetuned": f"{m_finetuned['precision']:.2f}%"},
        {"metric": "Macro Recall", "baseline": f"{m_baseline['recall']:.2f}%", "frozen": f"{m_frozen['recall']:.2f}%", "finetuned": f"{m_finetuned['recall']:.2f}%"},
        {"metric": "Macro F1 Score", "baseline": f"{m_baseline['f1_score']:.2f}%", "frozen": f"{m_frozen['f1_score']:.2f}%", "finetuned": f"{m_finetuned['f1_score']:.2f}%"},
        {"metric": "Training Duration", "baseline": "N/A", "frozen": "N/A", "finetuned": "N/A"},
        {"metric": "Inference Speed", "baseline": "N/A", "frozen": "N/A", "finetuned": "N/A"},
        {"metric": "Model Parameters", "baseline": "N/A", "frozen": "N/A", "finetuned": "N/A"},
        {"metric": "Model Checkpoint Size", "baseline": ckpt_base_size, "frozen": ckpt_froz_size, "finetuned": ckpt_fine_size},
        {"metric": "Embedding Vector Dimension", "baseline": "N/A", "frozen": "512D Vector" if ckpt_froz.exists() else "N/A", "finetuned": "512D Vector" if ckpt_fine.exists() else "N/A"},
        {"metric": "Checkpoint File", "baseline": ckpt_base.name if ckpt_base.exists() else "N/A", "frozen": ckpt_froz.name if ckpt_froz.exists() else "N/A", "finetuned": ckpt_fine.name if ckpt_fine.exists() else "N/A"},
    ]


    # Sample Image Prediction Comparison across all 3 models
    sample_img_path = PROJECT_ROOT / "datasets" / "eurosat" / "River" / "River_1.jpg"
    if not sample_img_path.exists():
        sample_imgs = list((PROJECT_ROOT / "datasets" / "eurosat" / "Forest").glob("*.jpg"))
        if len(sample_imgs) > 0:
            sample_img_path = sample_imgs[0]

    sample_rel_url = f"/datasets/{sample_img_path.relative_to(PROJECT_ROOT / 'datasets')}"
    sample_groundtruth = sample_img_path.parent.name

    from models.baseline_cnn import BaselineCNN
    from models.resnet18_model import ResNet18Model

    pred_base_cls, pred_base_conf = get_model_prediction(BaselineCNN, ckpt_base, sample_img_path)
    pred_froz_cls, pred_froz_conf = get_model_prediction(ResNet18Model, ckpt_froz, sample_img_path)
    pred_fine_cls, pred_fine_conf = get_model_prediction(ResNet18Model, ckpt_fine, sample_img_path)


    prediction_progression = {
        "image_url": sample_rel_url,
        "groundtruth": sample_groundtruth,
        "baseline": {"class": pred_base_cls, "confidence": pred_base_conf},
        "frozen": {"class": pred_froz_cls, "confidence": pred_froz_conf},
        "finetuned": {"class": pred_fine_cls, "confidence": pred_fine_conf},
    }

    return {
        "best_model": "Fine-Tuned ResNet18",
        "best_accuracy": m_finetuned["accuracy"],
        "accuracies": {
            "baseline": m_baseline["accuracy"],
            "frozen": m_frozen["accuracy"],
            "finetuned": m_finetuned["accuracy"],
        },
        "imp_baseline": imp_baseline,
        "imp_frozen": imp_frozen,
        "fine_tune_reason": "Unfreezing layer3, layer4, and fc residual blocks enabled deep spatial feature adaptation for satellite surface textures.",
        "table": metrics_table,
        "bar_chart_url": "/outputs/comparison/comparison_bar_chart.png",
        "matrices": {
            "baseline": "/outputs/baseline_cnn/confusion_matrix.png",
            "frozen": "/outputs/resnet18_frozen/confusion_matrix.png",
            "finetuned": "/outputs/resnet18_finetuned/confusion_matrix.png",
        },
        "curves": {
            "frozen_loss": "/outputs/resnet18_frozen/loss_curve.png",
            "frozen_acc": "/outputs/resnet18_frozen/accuracy_curve.png",
            "finetuned_loss": "/outputs/resnet18_finetuned/loss_curve.png",
            "finetuned_acc": "/outputs/resnet18_finetuned/accuracy_curve.png",
        },
        "prediction_progression": prediction_progression,
    }



def get_all_metrics() -> dict:
    """Default home dashboard summary metrics across models."""
    eurosat = get_dataset_info("eurosat")
    uc_merced = get_dataset_info("uc_merced")

    finetuned_acc = eurosat.get("training_accuracy", 0.0)
    finetuned_f1 = eurosat.get("f1_score", 0.0)

    baseline_acc = 0.0
    baseline_path = OUTPUTS_DIR / "baseline_cnn" / "metrics.json"
    if baseline_path.exists():
        try:
            with open(baseline_path, "r", encoding="utf-8") as f:
                d = json.load(f)
                val = d.get("accuracy", d.get("best_val_acc", d.get("final_val_acc", 0.0)))
                baseline_acc = round(float(val * 100 if val <= 1.0 else val), 2)
        except Exception:
            pass

    frozen_acc = 0.0
    frozen_path = OUTPUTS_DIR / "resnet18_frozen" / "metrics.json"
    if frozen_path.exists():
        try:
            with open(frozen_path, "r", encoding="utf-8") as f:
                d = json.load(f)
                val = d.get("accuracy", d.get("best_val_acc", d.get("final_val_acc", 0.0)))
                frozen_acc = round(float(val * 100 if val <= 1.0 else val), 2)
        except Exception:
            pass

    ucm_acc = uc_merced.get("training_accuracy", 0.0)

    # Check project_report_data.json fallback if available
    report_path = OUTPUTS_DIR / "project_report_data.json"
    if report_path.exists():
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                rep = json.load(f).get("report_values", {})
                if finetuned_acc == 0.0 and "test_accuracy" in rep:
                    finetuned_acc = float(rep["test_accuracy"])
                if ucm_acc == 0.0 and "uc_merced_accuracy" in rep:
                    ucm_acc = float(rep["uc_merced_accuracy"])
                if baseline_acc == 0.0 and "baseline_accuracy" in rep:
                    baseline_acc = float(rep["baseline_accuracy"])
                if frozen_acc == 0.0 and "frozen_resnet_accuracy" in rep:
                    frozen_acc = float(rep["frozen_resnet_accuracy"])
        except Exception:
            pass

    return {
        "eurosat": eurosat,
        "uc_merced": uc_merced,
        "resnet18_finetuned": {
            "accuracy": finetuned_acc,
            "f1_score": finetuned_f1,
        },
        "resnet18_frozen": {
            "accuracy": frozen_acc,
        },
        "baseline_cnn": {
            "accuracy": baseline_acc,
        },
        "best_accuracy": max(finetuned_acc, ucm_acc),
    }

