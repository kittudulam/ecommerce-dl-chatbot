"""Evaluation and Comparison Pipeline for LSTM and Bi-LSTM models."""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any

# Ensure project root is in Python's search path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)
import tensorflow as tf


def evaluate_models(
    models_dir: str = "saved_models",
    reports_dir: str = "reports",
) -> Dict[str, Any]:
    """Evaluate both models on validation set, generate confusion matrices and training plots."""
    models_path = Path(models_dir)
    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)

    print("=========================================================")
    print("STEP 1: Loading Models, Validation Data & Training History")
    print("=========================================================")
    
    # 1. Load pipeline metadata and classes
    with open(models_path / "pipeline_meta.json", "r") as f:
        meta = json.load(f)
    classes = meta["classes"]

    # 2. Load validation data
    val_data = np.load(models_path / "val_data.npz")
    X_val = val_data["X_val"]
    y_val = val_data["y_val"]
    y_true = np.argmax(y_val, axis=1)

    # 3. Load trained Keras models
    lstm_model = tf.keras.models.load_model(models_path / "lstm_model.keras")
    bilstm_model = tf.keras.models.load_model(models_path / "bilstm_model.keras")

    # 4. Load histories
    with open(models_path / "lstm_history.json", "r") as f:
        lstm_hist = json.load(f)
    with open(models_path / "bilstm_history.json", "r") as f:
        bilstm_hist = json.load(f)

    # =========================================================
    # STEP 2: Inferences & Latency Benchmarking
    # =========================================================
    print("STEP 2: Running Inferences & Benchmarking Latency...")
    # LSTM
    t0 = time.time()
    lstm_probs = lstm_model.predict(X_val, verbose=0)
    lstm_latency_ms = ((time.time() - t0) / len(X_val)) * 1000
    lstm_preds = np.argmax(lstm_probs, axis=1)

    # Bi-LSTM
    t0 = time.time()
    bilstm_probs = bilstm_model.predict(X_val, verbose=0)
    bilstm_latency_ms = ((time.time() - t0) / len(X_val)) * 1000
    bilstm_preds = np.argmax(bilstm_probs, axis=1)

    # =========================================================
    # STEP 3: Compute Performance Metrics
    # =========================================================
    lstm_acc = accuracy_score(y_true, lstm_preds)
    lstm_p_macro, lstm_r_macro, lstm_f1_macro, _ = precision_recall_fscore_support(
        y_true, lstm_preds, average="macro", zero_division=0
    )
    lstm_p_wt, lstm_r_wt, lstm_f1_wt, _ = precision_recall_fscore_support(
        y_true, lstm_preds, average="weighted", zero_division=0
    )

    bi_acc = accuracy_score(y_true, bilstm_preds)
    bi_p_macro, bi_r_macro, bi_f1_macro, _ = precision_recall_fscore_support(
        y_true, bilstm_preds, average="macro", zero_division=0
    )
    bi_p_wt, bi_r_wt, bi_f1_wt, _ = precision_recall_fscore_support(
        y_true, bilstm_preds, average="weighted", zero_division=0
    )

    # Print comparison table
    print("\n=========================================================")
    print("             MODEL PERFORMANCE COMPARISON                ")
    print("=========================================================")
    print(f"{'Metric':<25} | {'Standard LSTM':<16} | {'Bi-LSTM (Winner)':<16}")
    print("-" * 65)
    print(f"{'Validation Accuracy':<25} | {lstm_acc*100:>14.2f}% | {bi_acc*100:>14.2f}%")
    print(f"{'Macro Precision':<25} | {lstm_p_macro:>15.4f} | {bi_p_macro:>15.4f}")
    print(f"{'Macro Recall':<25} | {lstm_r_macro:>15.4f} | {bi_r_macro:>15.4f}")
    print(f"{'Macro F1-Score':<25} | {lstm_f1_macro:>15.4f} | {bi_f1_macro:>15.4f}")
    print(f"{'Weighted F1-Score':<25} | {lstm_f1_wt:>15.4f} | {bi_f1_wt:>15.4f}")
    print(f"{'Inference Latency':<25} | {lstm_latency_ms:>13.2f} ms | {bilstm_latency_ms:>13.2f} ms")
    print("=" * 65)

    print("\nDetailed Classification Report - Bi-LSTM:")
    print(classification_report(y_true, bilstm_preds, target_names=classes, zero_division=0))

    # =========================================================
    # STEP 4: Plot Training Curves (Loss & Accuracy)
    # =========================================================
    print("STEP 4: Plotting Training Curves...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy Plot
    axes[0].plot(lstm_hist.get("accuracy", []), label="LSTM Train Acc", linestyle="--", color="#1f77b4")
    axes[0].plot(lstm_hist.get("val_accuracy", []), label="LSTM Val Acc", color="#1f77b4", linewidth=2)
    axes[0].plot(bilstm_hist.get("accuracy", []), label="Bi-LSTM Train Acc", linestyle="--", color="#2ca02c")
    axes[0].plot(bilstm_hist.get("val_accuracy", []), label="Bi-LSTM Val Acc", color="#2ca02c", linewidth=2)
    axes[0].set_title("Training & Validation Accuracy Comparison", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="lower right")

    # Loss Plot
    axes[1].plot(lstm_hist.get("loss", []), label="LSTM Train Loss", linestyle="--", color="#1f77b4")
    axes[1].plot(lstm_hist.get("val_loss", []), label="LSTM Val Loss", color="#1f77b4", linewidth=2)
    axes[1].plot(bilstm_hist.get("loss", []), label="Bi-LSTM Train Loss", linestyle="--", color="#2ca02c")
    axes[1].plot(bilstm_hist.get("val_loss", []), label="Bi-LSTM Val Loss", color="#2ca02c", linewidth=2)
    axes[1].set_title("Training & Validation Loss (Cross-Entropy)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Categorical Cross-Entropy Loss")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    curves_file = reports_path / "training_curves_comparison.png"
    plt.savefig(curves_file, dpi=300)
    plt.close()
    print(f"-> Saved training curves plot: {curves_file}")

    # =========================================================
    # STEP 5: Plot Side-by-Side Confusion Matrices
    # =========================================================
    print("STEP 5: Generating Confusion Matrices...")
    cm_lstm = confusion_matrix(y_true, lstm_preds, labels=range(len(classes)))
    cm_bilstm = confusion_matrix(y_true, bilstm_preds, labels=range(len(classes)))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    sns.heatmap(
        cm_lstm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=axes[0],
        cbar=False,
    )
    axes[0].set_title(f"Standard LSTM Confusion Matrix\n(Accuracy: {lstm_acc*100:.1f}%)", fontweight="bold")
    axes[0].set_xlabel("Predicted Intent")
    axes[0].set_ylabel("True Intent")
    axes[0].tick_params(axis="x", rotation=45)

    sns.heatmap(
        cm_bilstm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=classes,
        yticklabels=classes,
        ax=axes[1],
        cbar=False,
    )
    axes[1].set_title(f"Bidirectional LSTM (Bi-LSTM) Confusion Matrix\n(Accuracy: {bi_acc*100:.1f}%)", fontweight="bold")
    axes[1].set_xlabel("Predicted Intent")
    axes[1].set_ylabel("True Intent")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    cm_file = reports_path / "confusion_matrices_comparison.png"
    plt.savefig(cm_file, dpi=300)
    plt.close()
    print(f"-> Saved confusion matrices plot: {cm_file}")

    # Save summary json
    summary = {
        "standard_lstm": {
            "accuracy": float(lstm_acc),
            "macro_precision": float(lstm_p_macro),
            "macro_recall": float(lstm_r_macro),
            "macro_f1": float(lstm_f1_macro),
            "weighted_f1": float(lstm_f1_wt),
            "latency_ms": float(lstm_latency_ms),
        },
        "bidirectional_lstm": {
            "accuracy": float(bi_acc),
            "macro_precision": float(bi_p_macro),
            "macro_recall": float(bi_r_macro),
            "macro_f1": float(bi_f1_macro),
            "weighted_f1": float(bi_f1_wt),
            "latency_ms": float(bilstm_latency_ms),
        },
        "winner": "Bidirectional_LSTM" if bi_f1_macro >= lstm_f1_macro else "Standard_LSTM",
    }
    with open(reports_path / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=========================================================")
    print("EVALUATION FINISHED! All comparison plots and summary saved in 'reports/'.")
    print("=========================================================")
    return summary


if __name__ == "__main__":
    evaluate_models()
