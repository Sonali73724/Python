"""
Streamlit Page: 01 Dataset Exploration
--------------------------------------
Interactive dataset explorer for EuroSAT satellite imagery and UC Merced aerial imagery.
Renders metadata, class distribution charts, and sample image grids.
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
import matplotlib.pyplot as plt
import numpy as np
from dashboard.services.metrics_service import get_dataset_page_data

st.set_page_config(page_title="Dataset Explorer - GeoAI Console", page_icon="📁", layout="wide")

from pages._sidebar import render_sidebar
render_sidebar()

st.title("📁 Dataset Exploration & Class Distributions")
st.markdown("Explore dataset metadata, class balance distributions, and sample patches for satellite and aerial imagery.")

# Session state active dataset sync
active_dataset = st.session_state.get("active_dataset", "eurosat")

dataset_choice = st.radio(
    "Select Active Dataset:",
    ["EuroSAT Sentinel-2 Satellite", "UC Merced Aerial Land Use"],
    index=0 if active_dataset == "eurosat" else 1,
    horizontal=True,
)

current_key = "eurosat" if "EuroSAT" in dataset_choice else "uc_merced"
st.session_state["active_dataset"] = current_key

data = get_dataset_page_data(current_key)
info = data["info"]
distribution = data["distribution"]
samples = data["sample_images"]

st.divider()

# Metadata Cards
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Dataset Name", info["name"])
c2.metric("Type", info["type"])
c3.metric("Total Images", f"{info['total_images']:,}")
c4.metric("Total Classes", info["total_classes"])
c5.metric("Resolution", info["resolution"])

st.divider()

# Class Distribution Section
st.subheader("📊 Class Balance Distribution")
if distribution["classes"]:
    col_chart, col_stats = st.columns([2, 1])

    with col_chart:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        y_pos = np.arange(len(distribution["classes"]))
        bars = ax.barh(y_pos, distribution["counts"], color="#2b5c8f", edgecolor="black", alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(distribution["classes"], fontsize=9, fontweight="bold")
        ax.invert_yaxis()
        ax.set_xlabel("Number of Image Samples", fontsize=10, fontweight="bold")
        ax.set_title(f"{info['name']} Class Count Distribution", fontsize=12, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.5)

        for bar in bars:
            width = bar.get_width()
            ax.annotate(
                f"{width:,}",
                xy=(width, bar.get_y() + bar.get_height() / 2),
                xytext=(5, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=8,
                fontweight="bold",
            )

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_stats:
        st.markdown("#### Distribution Statistics")
        st.markdown(f"**Total Dataset Count:** `{distribution['total_images']:,}` images")
        st.markdown(f"**Mean Images per Class:** `{int(np.mean(distribution['counts'])):,}`")
        st.markdown(f"**Source Sensor:** `{info['source']}`")
        st.markdown(f"**Data Split:** `{info['split_ratio']}`")
        st.info(info["description"])

st.divider()

# Sample Images Grid
st.subheader("🖼️ Class Sample Image Previews")
st.caption("One representative sample patch per Land Use / Land Cover class directory:")

if samples:
    cols_per_row = 5
    for i in range(0, len(samples), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j < len(samples):
                sample = samples[i + j]
                class_name = sample["class_name"]
                # Resolve local image path directly
                sample_img_path = PROJECT_ROOT / "datasets" / current_key / class_name / sample["file_name"]
                if sample_img_path.exists():
                    img = Image.open(sample_img_path)
                    col.image(img, caption=class_name, use_container_width=True)
                else:
                    col.warning(f"{class_name}\n(Preview Unavailable)")
else:
    st.warning("No sample images found in dataset directory.")
