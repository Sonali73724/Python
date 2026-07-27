"""
Metrics and Evaluation Utilities
---------------------------------
Functions for model evaluation, confusion matrix plotting,
and model comparison visualizations.
"""

import json
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def evaluate_model(model, dataloader, device):
    """
    Evaluates a PyTorch model on a given DataLoader.

    Args:
        model (torch.nn.Module): PyTorch model to evaluate.
        dataloader (torch.utils.data.DataLoader): Evaluation dataloader.
        device (str or torch.device): Device to run evaluation on.

    Returns:
        dict: Containing accuracy, precision, recall, f1_score, confusion_matrix, y_true, y_pred.
    """
    model.eval()
    model.to(device)
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "macro_precision": float(prec),
        "macro_recall": float(rec),
        "macro_f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def plot_confusion_matrix(cm, class_names, save_path=None, title="Confusion Matrix"):
    """
    Plots and optionally saves a heatmap confusion matrix.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(title, fontweight="bold", fontsize=14)
    plt.xlabel("Predicted Label", fontweight="bold")
    plt.ylabel("True Label", fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    
    return plt.gcf()


def plot_model_comparison(comparison_dict, save_path=None):
    """
    Plots a bar chart comparing Accuracy, Precision, Recall, and F1-Score across models.
    """
    models = list(comparison_dict.keys())
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]

    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, metric in enumerate(metrics):
        values = [comparison_dict[m].get(metric, 0.0) for m in models]
        rects = ax.bar(x + i * width, values, width, label=metric_labels[i])
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_ylabel("Score", fontweight="bold")
    ax.set_title("3-Model Accuracy & Multi-Metric Comparison", fontweight="bold", fontsize=14)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.15)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)

    return fig
