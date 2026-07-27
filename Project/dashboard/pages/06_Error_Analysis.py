"""
Streamlit Page: 06 Model Error Analysis
--------------------------------------
Model error analysis, top-5 misclassified samples, and confusion diagnosis.
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
from dashboard.services.error_analysis_service import get_error_analysis_data

st.set_page_config(page_title="Error Analysis - GeoAI Console", page_icon="🐞", layout="wide")
from pages._sidebar import render_sidebar
render_sidebar()

st.title("🐞 Model Error Analysis & Interpretability")
st.markdown("Isolate and diagnose the highest-confidence incorrect test predictions made by the Fine-Tuned ResNet18 model.")

ea_data = get_error_analysis_data()
summary = ea_data["summary"]
cards = ea_data["cards"]

st.divider()

# Error Analysis Summary Metrics
st.subheader("📊 Test Error Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Test Samples", summary.get("total_test_images", "N/A"))
c2.metric("Total Test Errors", summary.get("total_incorrect", "N/A"))
c3.metric("Overall Accuracy", summary.get("overall_accuracy", "N/A"))
c4.metric("Analyzed Top-5 Samples", summary.get("top5_count", "5 / 5"))

st.divider()

# Top-5 Misclassified Sample Cards
st.subheader("🔍 Top-5 Highest-Confidence Misclassifications")

if cards:
    cols = st.columns(min(5, len(cards)))
    for idx, card in enumerate(cards[:5]):
        with cols[idx]:
            st.markdown(f"#### Rank #{card['rank']}")
            st.error(f"**True:** `{card['ground_truth']}`\n\n**Pred:** `{card['predicted_class']}`")
            st.caption(f"**Confidence:** {card['confidence']}%")

            # Try displaying image
            img_path = PROJECT_ROOT / "outputs" / "error_analysis" / "top5_misclassified" / f"{card['image_name']}"
            if not img_path.exists():
                img_path = PROJECT_ROOT / "outputs" / "error_analysis" / "top5_misclassified" / f"image_{card['rank']}.png"

            if img_path.exists():
                st.image(Image.open(img_path), use_container_width=True)

            with st.expander("Root Cause Analysis"):
                st.write(card["reason"])
else:
    st.info("Error analysis data loading...")

st.divider()

# Error Grid Image Artifact
st.subheader("🖼️ Top-5 Misclassified Error Grid Artifact")
grid_path = PROJECT_ROOT / "outputs" / "error_analysis" / "top5_misclassified.png"
if grid_path.exists():
    st.image(Image.open(grid_path), caption="Top-5 Confident Error Grid Visualization", use_container_width=True)
else:
    st.warning("Error analysis grid artifact not found at `outputs/error_analysis/top5_misclassified.png`")
