"""
Streamlit Page: 03 Model Evaluation
-----------------------------------
Unified EuroSAT test set evaluation metrics, confusion matrix, and classification report.
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
from dashboard.services.metrics_service import get_merged_evaluation_data

st.set_page_config(page_title="Model Evaluation - GeoAI Console", page_icon="🎯", layout="wide")
from pages._sidebar import render_sidebar
render_sidebar()

st.title("🎯 Model Evaluation & Performance Analysis")
st.markdown("Comprehensive performance metrics on held-out test sets including confusion matrices and classification reports.")

merged_data = get_merged_evaluation_data()
eurosat_eval = merged_data["eurosat"]
ucm_eval = merged_data["uc_merced"]

tab1, tab2 = st.tabs(["🛰️ EuroSAT Primary Test Evaluation", "✈️ UC Merced Holdout Evaluation"])

# Tab 1: EuroSAT Evaluation
with tab1:
    st.subheader("Fine-Tuned ResNet18 (EuroSAT Test Set)")
    m = eurosat_eval["metrics"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall Accuracy", f"{m['accuracy']:.2f}%")
    c2.metric("Macro Precision", f"{m['precision']:.2f}%")
    c3.metric("Macro Recall", f"{m['recall']:.2f}%")
    c4.metric("Macro F1-Score", f"{m['f1_score']:.2f}%")

    st.divider()

    col_mat, col_rep = st.columns([1, 1])

    with col_mat:
        st.markdown("#### 📌 EuroSAT Confusion Matrix")
        matrix_path = PROJECT_ROOT / "outputs" / "resnet18_finetuned" / "confusion_matrix.png"
        if matrix_path.exists():
            st.image(Image.open(matrix_path), caption="Fine-Tuned ResNet18 Confusion Matrix", use_container_width=True)
        else:
            st.warning("Confusion matrix image not found at `outputs/resnet18_finetuned/confusion_matrix.png`")

    with col_rep:
        st.markdown("#### 📋 Detailed Classification Report")
        report_text = eurosat_eval.get("report_text", "")
        if report_text:
            st.code(report_text, language="text")
        else:
            st.info("Classification report available at `outputs/resnet18_finetuned/classification_report.txt`")

# Tab 2: UC Merced Holdout Evaluation
with tab2:
    st.subheader("ResNet18 Transfer Learning (UC Merced Holdout Set)")
    m_ucm = ucm_eval["metrics"]

    u1, u2, u3, u4 = st.columns(4)
    u1.metric("Holdout Accuracy", f"{m_ucm['accuracy']:.2f}%")
    u2.metric("Macro Precision", f"{m_ucm['precision']:.2f}%")
    u3.metric("Macro Recall", f"{m_ucm['recall']:.2f}%")
    u4.metric("Macro F1-Score", f"{m_ucm['f1_score']:.2f}%")

    st.divider()

    col_ucm_mat, col_ucm_rep = st.columns([1, 1])

    with col_ucm_mat:
        st.markdown("#### 📌 UC Merced Confusion Matrix")
        ucm_matrix_path = PROJECT_ROOT / "outputs" / "uc_merced" / "confusion_matrix.png"
        if ucm_matrix_path.exists():
            st.image(Image.open(ucm_matrix_path), caption="UC Merced ResNet18 Confusion Matrix", use_container_width=True)
        else:
            st.warning("UC Merced confusion matrix not found at `outputs/uc_merced/confusion_matrix.png`")

    with col_ucm_rep:
        st.markdown("#### 📋 UC Merced Classification Report")
        ucm_report_text = ucm_eval.get("report_text", "")
        if ucm_report_text:
            st.code(ucm_report_text, language="text")
        else:
            st.info("UC Merced classification report available at `outputs/uc_merced/classification_report.txt`")
