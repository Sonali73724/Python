# 🌍 GeoAI Research Console
### Deep Representation & Transfer Learning for Land Use / Land Cover (LULC) Classification & Change Detection

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13.0-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.60.0-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)

The **GeoAI Research Console** is a state-of-the-art computer vision platform designed for satellite and aerial remote sensing imagery analysis. It provides an end-to-end framework for evaluating deep learning architectures—ranging from scratch baseline CNNs to transfer learning backbones (ResNet18)—across Sentinel-2 satellite data (**EuroSAT**) and urban aerial imagery (**UC Merced**). 

The platform features an interactive 8-module Streamlit dashboard, a 512-D bi-temporal Siamese change detection engine, and automated error diagnostics.

---

## 🌟 Executive Summary & Key Results

| Model Architecture | Domain / Dataset | Strategy / Configuration | Test Accuracy | F1-Score | Status / Role |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Baseline CNN** | EuroSAT (Sentinel-2) | 3-Layer Conv Scratch | **78.05%** | 77.80% | Scratch Baseline |
| **Frozen ResNet18** | EuroSAT (Sentinel-2) | Fixed ImageNet Backbone | **89.83%** | 89.65% | Feature Extractor |
| **Fine-Tuned ResNet18** | EuroSAT (Sentinel-2) | Selective Layer Unfreezing | **97.11%** | **97.02%** | **SOTA Primary Model** |
| **ResNet18 Holdout** | UC Merced (Aerial) | Unfrozen Backbone | **96.19%** | 96.12% | Domain Generalization |

- **ROC-AUC (Siamese Change Detection)**: **`0.9966`** (Optimal threshold $\theta_{optimal} = 0.2698$)
- **Top EuroSAT Test Performance**: **`97.11%` Accuracy**, **`97.02%` F1-Score** achieved via Fine-Tuned ResNet18.

---

## 🚀 Key Features

### 1. 🛰️ Multi-Domain Remote Sensing Support
* **EuroSAT Dataset**: 27,000 Sentinel-2 satellite images across 10 Land Use / Land Cover (LULC) classes (`AnnualCrop`, `Forest`, `HerbaceousVegetation`, `Highway`, `Industrial`, `Pasture`, `PermanentCrop`, `Residential`, `River`, `SeaLake`).
* **UC Merced Land Use Dataset**: 2,100 high-resolution (0.3m) RGB aerial images across 21 urban land-use categories for external holdout evaluation.

### 2. 🔬 Tri-Model Benchmarking Engine
* **Baseline CNN**: 3-stage convolutional network trained from scratch to set performance baseline.
* **Frozen ResNet18**: Pre-trained ImageNet backbone serving as a static feature extractor with custom MLP classifier head.
* **Fine-Tuned ResNet18**: Advanced two-phase domain adaptation strategy. Layers `conv1` through `layer2` remain frozen, while `layer3`, `layer4`, and classifier head (`fc`) are unfrozen and trained at a lower learning rate ($1 \times 10^{-4}$).

### 3. 🔍 512-D Bi-Temporal Siamese Change Detection
* Leverages 512-dimensional latent feature embeddings from ResNet18 `avgpool` layer.
* Computes cosine distance between temporal image pairs:
  $$D(I_1, I_2) = 1 - \frac{f(I_1) \cdot f(I_2)}{\|f(I_1)\| \|f(I_2)\|}$$
* Employs **Youden's J Statistic** optimization ($J(\theta) = \text{TPR}(\theta) - \text{FPR}(\theta)$) to determine the mathematical decision threshold ($\theta = 0.2698$), yielding an **ROC-AUC of 0.9966**.

### 4. 🐞 Automated Top-5 Error Analysis
* Visual confidence diagnostics isolating edge cases and inter-class visual similarities (e.g., distinguishing `HerbaceousVegetation` vs. `Pasture`, or `Highway` vs. `Residential`).

### 5. 💻 Interactive Streamlit Research Console
* 8 dedicated interactive sub-pages supporting dataset exploration, loss/accuracy curves, confusion matrices, side-by-side comparative metrics, change detection file uploaders, error analysis cards, and theoretical documentation.

---

## 📂 Project Structure

```
project/
├── configs/                   # Configuration settings & dataset specifications
│   ├── class_labels.py        # Label mapping definitions
│   ├── config.py              # Hyperparameters, paths, device configuration
│   ├── dataset_info.py        # Dataset metadata & descriptions
│   └── uc_merced_config.py    # UC Merced configuration
├── dashboard/                 # Streamlit web application
│   ├── app.py                 # Main application entrypoint & executive metrics
│   ├── pages/                 # Interactive multi-page console modules
│   │   ├── 01_Dataset.py      # Dataset explorer & class distribution
│   │   ├── 02_Training.py     # Training progress & loss/accuracy progression
│   │   ├── 03_Evaluation.py   # Test set evaluation & confusion matrices
│   │   ├── 04_Comparison.py   # Tri-model benchmarking comparison
│   │   ├── 05_Change_Detection.py # Siamese change detection & pair upload
│   │   ├── 06_Error_Analysis.py   # Misclassification cards & diagnostics
│   │   ├── 07_UC_Merced.py    # Holdout generalization evaluation
│   │   └── 08_About.py        # Theoretical methodology & mathematical background
│   └── services/              # Business logic & inference services
│       ├── change_detection_service.py
│       ├── embedding_service.py
│       ├── error_analysis_service.py
│       ├── metrics_service.py
│       ├── model_manager.py
│       ├── similarity_service.py
│       └── uc_merced_service.py
├── datasets/                  # Remote sensing dataset storage
│   ├── eurosat/               # EuroSAT Sentinel-2 imagery
│   └── uc_merced_dataset.py   # UC Merced dataset loader
├── models/                    # PyTorch deep learning architecture definitions
│   ├── baseline_cnn.py        # 3-layer convolutional network
│   ├── resnet18_model.py      # EuroSAT ResNet18 transfer learning model
│   └── resnet18_ucmerced_model.py # UC Merced fine-tuned model
├── notebooks/                 # Sequential Jupyter notebooks (01 to 08)
├── checkpoints/               # Trained PyTorch model weights (`.pth`)
├── outputs/                   # Benchmark metrics, confusion matrices & JSON reports
├── utils/                     # Utility modules (data loading, metrics, evaluation pipeline)
│   ├── dataset.py
│   ├── evaluate_all.py        # Full automated pipeline evaluation script
│   ├── metrics.py
│   └── transforms.py
├── pyproject.toml             # Project dependency specification
└── README.md                  # Project documentation
```

---

## 🛠️ Installation & Setup

### Prerequisites
* Python **3.12+**
* PyTorch (supports CUDA GPU acceleration or CPU execution)

### Step 1: Clone & Navigate to Repository
```bash
git clone https://github.com/your-username/sonali-project.git
cd sonali-project
```

### Step 2: Set Up Virtual Environment & Dependencies

Using standard `venv`:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r pyproject.toml  # or pip install torch torchvision streamlit pandas numpy matplotlib seaborn scikit-learn opencv-python pillow tqdm
```

Using `uv` (recommended):
```bash
uv sync
source .venv/bin/activate
```

---

## 🚀 Running the Project

### 1. Launch Interactive Streamlit Dashboard
To launch the full interactive web application:
```bash
streamlit run dashboard/app.py
```
Open your browser and navigate to `http://localhost:8501`.

### 2. Execute Full Evaluation & Benchmarking Pipeline
To re-evaluate all models, regenerate metrics JSONs, confusion matrices, and model comparison reports:
```bash
python utils/evaluate_all.py
```

### 3. Run Notebook Pipeline
The `notebooks/` directory contains step-by-step Jupyter notebooks for exploratory analysis and training:
```bash
jupyter lab notebooks/
```

---

## 📊 Detailed Benchmark Results

### 1. EuroSAT Test Set Performance (2,700 Test Samples)

| Metric | Baseline CNN | Frozen ResNet18 | Fine-Tuned ResNet18 |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 78.05% | 89.83% | **97.11%** |
| **Macro F1-Score** | 77.80% | 89.65% | **97.02%** |
| **Macro Precision** | 78.20% | 90.10% | **97.15%** |
| **Macro Recall** | 77.95% | 89.70% | **97.05%** |
| **Test Loss** | 0.6241 | 0.3120 | **0.0892** |

### 2. Generalization Performance on UC Merced Holdout

* **Holdout Test Accuracy**: **`96.19%`**
* Demonstrates robust cross-domain generalization from Sentinel-2 satellite images (10m–60m resolution) to ultra-high-resolution aerial imagery (0.3m).

---

## 🔬 Technical Methodology & Mathematical Formulation

### 1. Domain Adaptation & Transfer Learning
Satellite imagery differs fundamentally from ImageNet datasets due to top-down aerial perspectives, non-object-centric spatial layouts, and rich texture frequencies. 

```
[ Input Image (224x224x3) ]
           │
 ┌─────────┴─────────┐
 │ conv1..layer2     │ ──► Frozen (ImageNet weights preserved)
 └─────────┬─────────┘
 ┌─────────┴─────────┐
 │ layer3..layer4    │ ──► Fine-Tuned (lr = 1e-4)
 └─────────┬─────────┘
 ┌─────────┴─────────┐
 │ Linear Classifier │ ──► Trained (256 -> Num Classes)
 └───────────────────┘
```

### 2. Siamese Cosine Distance & Change Threshold
Given two temporal satellite patches $I_1$ and $I_2$, feature representations $f(I_1), f(I_2) \in \mathbb{R}^{512}$ are extracted from the bottleneck embedding space. Cosine similarity and distance are computed as:

$$\text{Sim}(I_1, I_2) = \frac{f(I_1) \cdot f(I_2)}{\|f(I_1)\| \|f(I_2)\|}, \quad D(I_1, I_2) = 1 - \text{Sim}(I_1, I_2)$$

Using Youden's J statistic over ROC space:
$$J(\theta) = \text{Sensitivity}(\theta) + \text{Specificity}(\theta) - 1$$
$$\theta_{optimal} = \arg\max_\theta J(\theta) = 0.2698$$

Pairs with $D(I_1, I_2) > 0.2698$ are flagged as structural land-use changes.

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 🤝 Acknowledgments

* **EuroSAT Dataset**: European Space Agency (ESA) Sentinel-2 satellite dataset.
* **UC Merced Land Use Dataset**: USGS National Map Urban Area Imagery collection.
* **PyTorch & Torchvision**: Deep learning framework and pre-trained backbone models.

## Dataset
The original EuroSAT and UC Merced datasets are not included in this repository because of their large size.
Please download the datasets from their official sources and place them in the appropriate dataset directory before running the project.

## Model Checkpoint
The trained model checkpoint (.pth) is not included because it exceeds GitHub's 100 MB file size limit.
The project can be retrained using the provided notebooks and source code.

## Demo Video
The demo video is included in this repository.
If GitHub cannot preview the video due to its size, please download the file and play it locally.
