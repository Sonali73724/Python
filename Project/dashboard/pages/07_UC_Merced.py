"""
Streamlit Page: 07 UC Merced Holdout Evaluation
-----------------------------------------------
External holdout generalization metrics on UC Merced Land Use dataset (21 urban classes).
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
from dashboard.services.uc_merced_service import get_uc_merced_metrics
from configs.dataset_info import get_dataset_info

st.set_page_config(page_title="UC Merced Holdout - GeoAI Console", page_icon="✈️", layout="wide")
from pages._sidebar import render_sidebar
render_sidebar()
st.title("✈️ UC Merced Land Use Holdout Evaluation")
st.markdown("Independent domain generalization test evaluating ResNet18 transfer learning on high-resolution aerial imagery (21 urban land-use classes).")

metrics = get_uc_merced_metrics()
info = get_dataset_info("uc_merced")

st.divider()

# Overview & Metrics Cards
st.subheader("📊 UC Merced Holdout Performance Metrics")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Holdout Accuracy", f"{metrics.get('accuracy_pct', 0.0):.2f}%")
c2.metric("Macro Precision", f"{metrics.get('precision', 0.0):.2f}%")
c3.metric("Macro Recall", f"{metrics.get('recall', 0.0):.2f}%")
c4.metric("Macro F1-Score", f"{metrics.get('f1_score', 0.0):.2f}%")
c5.metric("Total Images", f"{metrics.get('total_images', 2100):,}")

st.divider()

# Dataset Overview & Domain Comparison
st.subheader("🏢 Dataset Metadata & Aerial Domain Profile")

col_info1, col_info2 = st.columns(2)

with col_info1:
    st.markdown("#### ✈️ Dataset Specifications")
    st.markdown(f"**Dataset Name:** `{info['name']}`")
    st.markdown(f"**Sensor Source:** `{info['source']}`")
    st.markdown(f"**Resolution:** `{info['resolution']}`")
    st.markdown(f"**Total Classes:** `{info['total_classes']} Classes`")
    st.markdown(f"**Fine-Tuned Checkpoint:** `{info['checkpoint']}`")

with col_info2:
    st.markdown("#### 🔬 Domain Differences: Satellite vs Aerial")
    st.info("""
    - **EuroSAT (Sentinel-2)**: Multispectral satellite patches (13 spectral bands, 10m spatial resolution) covering natural LULC classes.
    - **UC Merced**: RGB aerial imagery (0.3m ultra-high resolution) extracted from USGS National Map covering complex man-made urban structures.
    """)

st.divider()

# Confusion Matrix & Classification Report
col_mat, col_rep = st.columns([1, 1])

with col_mat:
    st.markdown("#### 📌 UC Merced Confusion Matrix")
    matrix_path = PROJECT_ROOT / "outputs" / "uc_merced" / "confusion_matrix.png"
    if matrix_path.exists():
        st.image(Image.open(matrix_path), caption="UC Merced 21-Class Confusion Matrix", use_container_width=True)
    else:
        st.warning("Confusion matrix not found at `outputs/uc_merced/confusion_matrix.png`")

with col_rep:
    st.markdown("#### 📋 UC Merced Classification Report")
    report_text = metrics.get("classification_report", "")
    if report_text:
        st.code(report_text, language="text")
    else:
        st.info("Classification report available at `outputs/uc_merced/classification_report.txt`")
