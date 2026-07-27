"""
Error Analysis Service Module
-----------------------------
Service layer responsible for loading EuroSAT Top-5 misclassified image analysis results
for rendering in the Flask GeoAI console dashboard.
Reads statistics directly from outputs/error_analysis/summary.json and cards from error_analysis.csv.
Returns safe 'N/A' fallbacks if summary.json or CSV outputs are missing or malformed.
"""

import csv
import json
import sys
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parent
DASHBOARD_DIR = SERVICE_DIR.parent
PROJECT_ROOT = DASHBOARD_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from evaluation.error_analysis import get_reason_placeholder
except ImportError:
    def get_reason_placeholder(gt: str, pred: str) -> str:
        return f"Inter-class visual spectral similarity between {gt} and {pred}."


OUTPUTS_EA_DIR = PROJECT_ROOT / "outputs" / "error_analysis"
CSV_PATH = OUTPUTS_EA_DIR / "error_analysis.csv"
SUMMARY_PATH = OUTPUTS_EA_DIR / "summary.json"
GRID_PATH = OUTPUTS_EA_DIR / "top5_misclassified.png"
MD_PATH = OUTPUTS_EA_DIR / "error_analysis.md"


def get_error_analysis_data() -> dict:
    """
    Retrieve error analysis metrics directly from summary.json and Top-5 misclassified samples
    from error_analysis.csv. If files are missing or malformed, returns N/A fallbacks.

    Returns:
        dict: Summary statistics, card metrics, and artifact file URLs.
    """
    # Safe default fallbacks when summary.json is missing or invalid
    summary = {
        "total_test_images": "N/A",
        "total_incorrect": "N/A",
        "overall_accuracy": "N/A",
        "top5_count": "N/A",
    }

    # 1. Read statistics directly from outputs/error_analysis/summary.json
    if SUMMARY_PATH.exists():
        try:
            with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
                s_data = json.load(f)
                if isinstance(s_data, dict):
                    # Format total_test_images
                    tot_img = s_data.get("total_test_images")
                    if isinstance(tot_img, (int, float)):
                        summary["total_test_images"] = f"{int(tot_img):,}"
                    elif tot_img is not None and str(tot_img).strip() != "":
                        summary["total_test_images"] = str(tot_img)

                    # Format total_incorrect
                    tot_inc = s_data.get("total_incorrect")
                    if isinstance(tot_inc, (int, float)):
                        summary["total_incorrect"] = f"{int(tot_inc):,}"
                    elif tot_inc is not None and str(tot_inc).strip() != "":
                        summary["total_incorrect"] = str(tot_inc)

                    # Format overall_accuracy
                    acc = s_data.get("overall_accuracy")
                    if acc is not None and str(acc).strip() != "":
                        if isinstance(acc, (int, float)):
                            summary["overall_accuracy"] = f"{acc * 100 if acc <= 1.0 else acc:.2f}%"
                        else:
                            summary["overall_accuracy"] = str(acc)

                    # Format top5_count
                    top5_c = s_data.get("top5_count")
                    if top5_c is not None and str(top5_c).strip() != "":
                        top5_str = str(top5_c).strip()
                        if top5_str.isdigit():
                            summary["top5_count"] = f"{top5_str} / 5"
                        else:
                            summary["top5_count"] = top5_str
        except Exception as err:
            print(f"[Warning] Exception while reading {SUMMARY_PATH}: {err}")

    # 2. Read Top-5 misclassified samples from error_analysis.csv
    cards = []
    if CSV_PATH.exists():
        try:
            with open(CSV_PATH, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for idx, row in enumerate(reader, start=1):
                    img_name = row.get("Image Name") or "N/A"
                    gt = row.get("Ground Truth") or "N/A"
                    pred = row.get("Predicted Class") or "N/A"
                    conf = row.get("Confidence") or "N/A"

                    # Determine image URL dynamically based on file existence
                    image_url = None
                    if img_name != "N/A":
                        img_path = OUTPUTS_EA_DIR / "top5_misclassified" / img_name
                        if img_path.exists():
                            image_url = f"/outputs/error_analysis/top5_misclassified/{img_name}"
                        else:
                            numbered_path = OUTPUTS_EA_DIR / "top5_misclassified" / f"image_{idx}.png"
                            if numbered_path.exists():
                                image_url = f"/outputs/error_analysis/top5_misclassified/image_{idx}.png"

                    cards.append({
                        "rank": idx,
                        "image_name": img_name,
                        "image_url": image_url,
                        "ground_truth": gt,
                        "predicted_class": pred,
                        "confidence": conf,
                        "reason": get_reason_placeholder(gt, pred) if gt != "N/A" and pred != "N/A" else "Evaluation data unavailable.",
                    })
        except Exception as err:
            print(f"[Warning] Exception while reading {CSV_PATH}: {err}")

    has_data = len(cards) > 0

    return {
        "summary": summary,
        "cards": cards,
        "has_data": has_data,
        "grid_url": "/outputs/error_analysis/top5_misclassified.png" if (GRID_PATH.exists() and has_data) else None,
        "csv_url": "/outputs/error_analysis/error_analysis.csv" if (CSV_PATH.exists() and has_data) else None,
        "md_url": "/outputs/error_analysis/error_analysis.md" if (MD_PATH.exists() and has_data) else None,
    }
