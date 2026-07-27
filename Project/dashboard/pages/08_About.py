"""
Streamlit Page: 08 About & Technical Methodology
------------------------------------------------
Technical documentation, system architecture, remote sensing methodology, and transfer learning math.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st

st.set_page_config(page_title="About & Methodology - GeoAI Console", page_icon="📖", layout="wide")
from pages._sidebar import render_sidebar
render_sidebar()
st.title("📖 About & Technical Methodology")
st.markdown("Detailed documentation on deep representation learning, remote sensing domain characteristics, and Siamese temporal change detection.")

st.divider()

# Section 1: Project Overview
st.subheader("🌐 GeoAI Research Console Platform")
st.markdown("""
The **GeoAI Research Console** is an advanced computer vision benchmarking platform engineered to evaluate deep representation learning for Land Use / Land Cover (LULC) classification and temporal change detection across satellite and aerial remote sensing imagery.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🛰️ EuroSAT Sentinel-2 Satellite Domain")
    st.markdown("""
    - **Source**: European Space Agency (ESA) Sentinel-2 Satellite.
    - **Spectral Bands**: 13 Spectral Bands (B1 - B12) including Visible, NIR, and SWIR.
    - **Spatial Resolution**: 10m - 60m per pixel.
    - **Land Use Classes**: 10 Classes (`AnnualCrop`, `Forest`, `HerbaceousVegetation`, `Highway`, `Industrial`, `Pasture`, `PermanentCrop`, `Residential`, `River`, `SeaLake`).
    """)

with col2:
    st.markdown("#### ✈️ UC Merced Aerial Domain")
    st.markdown("""
    - **Source**: USGS National Map Urban Area Imagery.
    - **Spectral Bands**: RGB 3-channel optical.
    - **Spatial Resolution**: 0.3m ultra-high resolution per pixel.
    - **Urban Land Use Classes**: 21 Classes (`agricultural`, `airplane`, `baseballdiamond`, `beach`, `buildings`, `chaparral`, `densebuilding`, `forest`, `freeway`, `golfcourse`, `harbor`, `intersection`, `mediumresidential`, `mobilehomepark`, `overpass`, `parkinglot`, `river`, `runway`, `sparsebuilding`, `storagetanks`, `tenniscourt`).
    """)

st.divider()

# Section 2: Transfer Learning Strategy
st.subheader("🔬 Deep Representation Transfer Learning Strategy")
st.markdown("""
Natural image datasets like ImageNet feature object-centric RGB compositions, whereas satellite remote sensing imagery exhibits texture-rich, overhead, non-object-centric land patterns.

To effectively adapt a pre-trained **ResNet18** architecture:
1. **Backbone Feature Extractor**: Freeze `conv1` through `layer2` to preserve generic edge and color filters.
2. **Selective Unfreezing**: Unfreeze `layer3`, `layer4`, and classification head (`fc`) with a reduced learning rate ($1 \times 10^{-4}$).
3. **512-D Latent Space Embedding**: Extract the 512-dimensional output from `avgpool` for Siamese feature distance computation.
""")

st.divider()

# Section 3: Siamese Change Detection Math
st.subheader("📐 Siamese Feature Distance & Youden J Threshold Math")

st.latex(r"""
\text{Sim}(I_1, I_2) = \frac{f(I_1) \cdot f(I_2)}{\|f(I_1)\| \|f(I_2)\|}
""")

st.latex(r"""
D(I_1, I_2) = 1 - \text{Sim}(I_1, I_2)
""")

st.latex(r"""
J(\theta) = \text{TPR}(\theta) - \text{FPR}(\theta) \implies \theta_{optimal} = \arg\max_\theta J(\theta)
""")

st.markdown("""
- **Cosine Distance ($D$)**: Measures angular displacement in 512-D feature space between bi-temporal satellite image pairs $I_1$ and $I_2$.
- **Youden's J Statistic ($J$)**: Maximizes the trade-off between True Positive Rate and False Positive Rate on bi-temporal test pairs to compute optimal decision threshold $\theta = 0.2698$, yielding an **ROC-AUC of 0.9966**.
""")

st.divider()

# Section 4: System Architecture
st.subheader("💻 Modular Streamlit Architecture")
st.markdown("""
The web application is structured cleanly using modular Streamlit pages and service layers:
```
dashboard/
├── app.py                     # Main Streamlit Landing Page & Executive Dashboard
└── pages/
    ├── 01_Dataset.py          # Interactive Dataset Explorer & Class Distributions
    ├── 02_Training.py         # Loss & Accuracy Progression Curves
    ├── 03_Evaluation.py       # Unified EuroSAT Test Evaluation & Confusion Matrix
    ├── 04_Comparison.py       # Tri-Model Comparative Benchmarking
    ├── 05_Change_Detection.py # Siamese Bi-Temporal Change Detection & File Uploaders
    ├── 06_Error_Analysis.py   # Top-5 Misclassified Sample Cards & Root Cause Diagnosis
    ├── 07_UC_Merced.py        # External UC Merced Holdout Evaluation
    └── 08_About.py            # Technical Methodology & Theoretical Background
```
""")
