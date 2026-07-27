"""
UC Merced Service Module
-----------------------
Service layer responsible for loading UC Merced evaluation metrics, classification report,
and confusion matrix, with safe fallbacks so the dashboard never crashes.
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

try:
    from evaluation.evaluate_uc_merced import evaluate_uc_merced
except ImportError:
    evaluate_uc_merced = None


OUTPUTS_UCM_DIR = PROJECT_ROOT / "outputs" / "uc_merced"

# Default fallback metrics if evaluation output file is missing (strictly 0.00 / 0)
DEMO_UCM_METRICS = {
    "accuracy": 0.00,
    "macro_precision": 0.00,
    "macro_recall": 0.00,
    "macro_f1": 0.00,
    "precision": 0.00,
    "recall": 0.00,
    "f1_score": 0.00,
    "total_images": 0,
}





def get_uc_merced_metrics() -> dict:
    """
    Read UC Merced evaluation metrics from outputs/uc_merced/metrics.json.
    Includes classification report text and confusion matrix path if present.
    Executes evaluation if outputs do not exist yet.

    Returns:
        dict: UC Merced evaluation metrics, report text, and matrix URL.
    """
    metrics_path = OUTPUTS_UCM_DIR / "metrics.json"
    report_path = OUTPUTS_UCM_DIR / "classification_report.txt"
    matrix_path = OUTPUTS_UCM_DIR / "confusion_matrix.png"

    # Run evaluation if metrics.json is not present and evaluator is available
    if not metrics_path.exists() and evaluate_uc_merced is not None:
        try:
            evaluate_uc_merced()
        except Exception as err:
            print(f"[Warning] UC Merced evaluation run failed: {err}")

    res = dict(DEMO_UCM_METRICS)

    try:
        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    res.update(data)
                    # Convert metrics to formatted percentages for UI display
                    acc = res.get("accuracy", 0)
                    res["accuracy_pct"] = round(acc * 100, 2) if acc <= 1.0 else acc

                    prec = res.get("macro_precision", 0)
                    res["precision"] = round(prec * 100, 2) if prec <= 1.0 else prec

                    rec = res.get("macro_recall", 0)
                    res["recall"] = round(rec * 100, 2) if rec <= 1.0 else rec

                    f1 = res.get("macro_f1", 0)
                    res["f1_score"] = round(f1 * 100, 2) if f1 <= 1.0 else f1

    except Exception as err:
        print(f"[Warning] Failed to load UC Merced metrics from {metrics_path}: {err}")

    # Check for classification report text
    report_text = ""
    if report_path.exists():
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                report_text = f.read()
        except Exception:
            report_text = ""
    res["classification_report"] = report_text

    # Check if confusion matrix image exists
    res["matrix_url"] = (
        "/outputs/uc_merced/confusion_matrix.png"
        if matrix_path.exists()
        else None
    )

    return res
