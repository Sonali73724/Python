"""
Streamlit Page: 05 Siamese Temporal Change Detection
---------------------------------------------------
Interactive bi-temporal satellite image upload & land-use change detection interface.
"""

import sys
import time
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st
from dashboard.services.change_detection_service import detect_change

st.set_page_config(page_title="Change Detection - GeoAI Console", page_icon="🔍", layout="wide")
from pages._sidebar import render_sidebar
render_sidebar()

st.title("🔍 Siamese Temporal Change Detection")
st.markdown(
    "Extract 512-dimensional latent feature vectors using fine-tuned ResNet18 backbone, compute "
    "Cosine Similarity ($1 - D$), and apply Youden J optimal decision thresholding ($\theta = 0.2698$) "
    "to detect structural land-use changes."
)

st.divider()

# Controls & Upload Section
col_upload1, col_upload2 = st.columns(2)

with col_upload1:
    st.subheader("🖼️ Pre-Change Image ($T_1$)")
    file_before = st.file_uploader(
        "Upload Before Satellite / Aerial Image",
        type=["png", "jpg", "jpeg", "tif", "tiff", "webp"],
        key="before_uploader",
        help="Select initial baseline temporal satellite patch.",
    )
    if file_before:
        st.image(file_before, caption="Pre-Change Image (T1)", use_container_width=True)

with col_upload2:
    st.subheader("🖼️ Post-Change Image ($T_2$)")
    file_after = st.file_uploader(
        "Upload After Satellite / Aerial Image",
        type=["png", "jpg", "jpeg", "tif", "tiff", "webp"],
        key="after_uploader",
        help="Select subsequent temporal satellite patch.",
    )
    if file_after:
        st.image(file_after, caption="Post-Change Image (T2)", use_container_width=True)

st.divider()

col_opt1, col_opt2 = st.columns(2)

with col_opt1:
    active_dataset = st.session_state.get("active_dataset", "eurosat")
    dataset_type = st.selectbox(
        "Select Model Domain Dataset:",
        ["EuroSAT Sentinel-2 Satellite", "UC Merced Aerial Imagery"],
        index=0 if active_dataset == "eurosat" else 1,
    )
    dataset_key = "eurosat" if "EuroSAT" in dataset_type else "uc_merced"

with col_opt2:
    custom_threshold = st.slider(
        "Cosine Distance Threshold ($\theta$):",
        min_value=0.0,
        max_value=1.0,
        value=0.2698,
        step=0.01,
        help="Default is optimal Youden J threshold (θ = 0.2698) derived from ROC-AUC evaluation.",
    )

run_button = st.button("🚀 Run Temporal Change Detection", type="primary", use_container_width=True)

st.divider()

if run_button:
    if not file_before or not file_after:
        st.error("⚠️ Please upload both 'Before' ($T_1$) and 'After' ($T_2$) image patches before running change detection.")
    else:
        with st.spinner("Extracting 512-D latent feature embeddings and computing spatial heatmaps..."):
            uploads_dir = PROJECT_ROOT / "uploads"
            uploads_dir.mkdir(parents=True, exist_ok=True)

            timestamp = int(time.time() * 1000)
            path_before = uploads_dir / f"{timestamp}_before_{file_before.name}"
            path_after = uploads_dir / f"{timestamp}_after_{file_after.name}"

            with open(path_before, "wb") as f:
                f.write(file_before.getbuffer())

            with open(path_after, "wb") as f:
                f.write(file_after.getbuffer())

            try:
                res = detect_change(
                    path_before,
                    path_after,
                    threshold=custom_threshold,
                    dataset=dataset_key,
                )

                st.subheader("🎯 Change Detection Analysis Results")

                # Metrics Banner
                if res["changed"]:
                    st.error(f"🚨 **STRUCTURAL LAND-USE CHANGE DETECTED** — {res['status']}")
                else:
                    st.success(f"✅ **NO SIGNIFICANT CHANGE DETECTED** — {res['status']}")

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Pre-Change Class ($T_1$)", f"{res['before_class']}", f"{res['confidence_before']}% Conf")
                m2.metric("Post-Change Class ($T_2$)", f"{res['after_class']}", f"{res['confidence_after']}% Conf")
                m3.metric("Cosine Similarity", f"{res['similarity']:.4f}")
                m4.metric("Cosine Distance", f"{1.0 - res['similarity']:.4f}")
                m5.metric("Decision Threshold", f"{res['threshold']:.4f}")

                st.divider()

                # Visual Output Artifacts
                st.subheader("📊 Spatial Change Visualizations & Heatmaps")

                vis_col1, vis_col2, vis_col3 = st.columns(3)

                chart_path = PROJECT_ROOT / "outputs" / "change_detection" / "confidence_chart.png"
                diff_path = PROJECT_ROOT / "outputs" / "change_detection" / "difference.png"
                overlay_path = PROJECT_ROOT / "outputs" / "change_detection" / "heatmap_overlay.png"

                with vis_col1:
                    st.markdown("#### 📊 Prediction Confidence Comparison")
                    if chart_path.exists():
                        st.image(Image.open(chart_path), use_container_width=True)

                with vis_col2:
                    st.markdown("#### 🗺️ Difference Map")
                    if diff_path.exists():
                        st.image(Image.open(diff_path), use_container_width=True)

                with vis_col3:
                    st.markdown("#### 🔥 Heatmap Overlay")
                    if overlay_path.exists():
                        st.image(Image.open(overlay_path), use_container_width=True)

            except Exception as err:
                st.error(f"❌ Error performing change detection: {err}")

# Show default change detection artifacts if already present
else:
    st.subheader("📌 Pre-Generated Change Detection Artifacts")
    roc_path = PROJECT_ROOT / "outputs" / "change_detection" / "roc_curve.png"
    overlay_path = PROJECT_ROOT / "outputs" / "change_detection" / "heatmap_overlay.png"

    if roc_path.exists() or overlay_path.exists():
        c_roc, c_ov = st.columns(2)
        with c_roc:
            if roc_path.exists():
                st.image(Image.open(roc_path), caption="Change Detection ROC Curve (AUC = 0.9966)", use_container_width=True)
        with c_ov:
            if overlay_path.exists():
                st.image(Image.open(overlay_path), caption="Bi-Temporal Change Heatmap Overlay", use_container_width=True)
