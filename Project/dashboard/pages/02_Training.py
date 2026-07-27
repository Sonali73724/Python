"""
Streamlit Page: 02 Training Progress
-----------------------------------
Training progress, loss curves, accuracy history, and hyperparameter config.
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
from dashboard.services.metrics_service import get_training_page_data

st.set_page_config(page_title="Training Progress - GeoAI Console", page_icon="📈", layout="wide")
from pages._sidebar import render_sidebar
render_sidebar()

st.title("📈 Model Training Progress & Optimization")
st.markdown("Monitor training loss reduction, validation accuracy progression, and hyperparameter settings.")

active_dataset = st.session_state.get("active_dataset", "eurosat")

dataset_choice = st.radio(
    "Select Model Architecture Context:",
    ["Fine-Tuned ResNet18 (EuroSAT)", "ResNet18 Transfer Learning (UC Merced)"],
    index=0 if active_dataset == "eurosat" else 1,
    horizontal=True,
)

current_key = "eurosat" if "EuroSAT" in dataset_choice else "uc_merced"
st.session_state["active_dataset"] = current_key

data = get_training_page_data(current_key)
info = data["info"]

st.divider()

# Hyperparameter Configuration Box
st.subheader("⚙️ Training Hyperparameters & Configuration")
hp_col1, hp_col2, hp_col3, hp_col4, hp_col5 = st.columns(5)
hp_col1.metric("Model Architecture", info["model_name"])
hp_col2.metric("Epochs Trained", info["epochs"])
hp_col3.metric("Optimizer", info["optimizer"])
hp_col4.metric("Learning Rate", info["learning_rate"])
hp_col5.metric("Embedding Dim", info["embedding_dim"])

st.divider()

# Training & Validation Curves
st.subheader("📊 Loss & Accuracy Progression Curves")

col_loss, col_acc = st.columns(2)

if current_key == "uc_merced":
    loss_img_path = PROJECT_ROOT / "outputs" / "uc_merced" / "loss_curve.png"
    acc_img_path = PROJECT_ROOT / "outputs" / "uc_merced" / "accuracy_curve.png"
else:
    loss_img_path = PROJECT_ROOT / "outputs" / "resnet18_finetuned" / "loss_curve.png"
    acc_img_path = PROJECT_ROOT / "outputs" / "resnet18_finetuned" / "accuracy_curve.png"

with col_loss:
    st.markdown("#### 📉 Training vs Validation Loss Curve")
    if loss_img_path.exists():
        st.image(Image.open(loss_img_path), caption="Training & Validation Loss Reduction", use_container_width=True)
    else:
        st.info("Loss curve plot artifact available at `outputs/resnet18_finetuned/training_history.png`")

with col_acc:
    st.markdown("#### 🎯 Validation Accuracy History Curve")
    if acc_img_path.exists():
        st.image(Image.open(acc_img_path), caption="Epoch Accuracy Progression", use_container_width=True)
    else:
        st.info("Accuracy curve plot artifact available at `outputs/resnet18_finetuned/training_history.png`")

st.divider()

# Combined Training History Artifact
st.subheader("🖼️ Training History Plot Artifact")
combined_history_path = (
    PROJECT_ROOT / "outputs" / ("uc_merced" if current_key == "uc_merced" else "resnet18_finetuned") / "training_history.png"
)
if combined_history_path.exists():
    st.image(Image.open(combined_history_path), caption="Combined Training & Validation History", use_container_width=True)
else:
    st.caption("Training history plots generated during training phase.")
