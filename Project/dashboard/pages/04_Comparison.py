"""
Streamlit Page: 04 Comparative Benchmarking
-------------------------------------------
3-Model Comparative Benchmarking (Baseline CNN vs Frozen ResNet18 vs Fine-Tuned ResNet18).
"""

import sys
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st
import pandas as pd
from dashboard.services.metrics_service import get_comparison_page_data

st.set_page_config(page_title="Model Comparison - GeoAI Console", page_icon="⚖️", layout="wide")
from pages._sidebar import render_sidebar
render_sidebar()

st.title("⚖️ Tri-Model Comparative Benchmarking")
st.markdown("Side-by-side performance evaluation across Baseline Custom CNN, Frozen ResNet18 Feature Extractor, and Unfrozen Fine-Tuned ResNet18.")

comp_data = get_comparison_page_data()

st.divider()

# Core Metric Improvement Cards
st.subheader("🚀 Performance Gain Summary")
c1, c2, c3 = st.columns(3)

finetuned_acc = comp_data.get("best_accuracy", 0.0)
imp_baseline = comp_data.get("imp_baseline", 0.0)
imp_frozen = comp_data.get("imp_frozen", 0.0)

c1.metric("Fine-Tuned ResNet18", f"{finetuned_acc:.2f}%", help="Primary production model accuracy")
c2.metric("Gain Over Baseline CNN", f"+{imp_baseline:.2f}%", delta=f"{imp_baseline:.2f}%", help="Accuracy gain over scratch CNN model")
c3.metric("Gain Over Frozen ResNet18", f"+{imp_frozen:.2f}%", delta=f"{imp_frozen:.2f}%", help="Accuracy gain from unfreezing layer3 & layer4")

st.divider()

# Comparative Metrics Table
st.subheader("📋 Comprehensive Model Comparison Table")

table_rows = comp_data.get("table", [])
if table_rows:
    df = pd.DataFrame(table_rows)
    df.columns = ["Metric", "Baseline Custom CNN", "Frozen ResNet18", "Fine-Tuned ResNet18"]
    st.dataframe(df, use_container_width=True, hide_index=True)

st.divider()

# Comparison Chart Image Artifact
st.subheader("📊 Comparative Bar Charts & Confusion Matrices")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("#### 📈 Model Comparison Chart")
    comp_chart_path = PROJECT_ROOT / "outputs" / "comparison" / "model_comparison.png"
    if comp_chart_path.exists():
        st.image(Image.open(comp_chart_path), caption="Tri-Model Comparative Accuracy & F1-Score Bar Chart", use_container_width=True)
    else:
        st.info("Comparison chart artifact available at `outputs/comparison/model_comparison.png`")

with chart_col2:
    st.markdown("#### 🎯 Baseline CNN Confusion Matrix")
    base_matrix_path = PROJECT_ROOT / "outputs" / "baseline_cnn" / "confusion_matrix.png"
    if base_matrix_path.exists():
        st.image(Image.open(base_matrix_path), caption="Baseline Custom CNN Confusion Matrix", use_container_width=True)
    else:
        st.info("Baseline confusion matrix available at `outputs/baseline_cnn/confusion_matrix.png`")

st.divider()

# Sample Image Prediction Comparison across 3 models
st.subheader("📸 Sample Prediction Comparison Across Models")
st.caption("Inspect live prediction outputs for a representative test sample across all 3 models:")

prog = comp_data.get("prediction_progression", {})
if prog:
    gt = prog.get("groundtruth", "N/A")
    base_p = prog.get("baseline", {})
    froz_p = prog.get("frozen", {})
    fine_p = prog.get("finetuned", {})

    st.markdown(f"**Ground Truth Class:** `{gt}`")
    st.markdown(f"**Baseline CNN Prediction:** `{base_p.get('class', 'N/A')}` ({base_p.get('confidence', 0)}% Conf)")
    st.markdown(f"**Frozen ResNet18 Prediction:** `{froz_p.get('class', 'N/A')}` ({froz_p.get('confidence', 0)}% Conf)")
    st.markdown(f"**Fine-Tuned ResNet18 Prediction:** `{fine_p.get('class', 'N/A')}` ({fine_p.get('confidence', 0)}% Conf)")
