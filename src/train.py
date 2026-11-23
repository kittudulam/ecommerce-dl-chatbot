"""Training pipeline for LSTM and Bi-LSTM E-commerce Intent Classifiers."""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any

# Ensure project root is always in Python's search path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from src.preprocessor import load_intents_data, IntentDataPipeline
from src.models import build_lstm_model, build_bilstm_model


def train_models(
    data_path: str = "data/ecommerce_intents.json",
    save_dir: str = "saved_models",
    epochs: int = 70,
    batch_size: int = 8,
) -> Dict[str, Any]:
    """Train both LSTM and Bi-LSTM models and save artifacts."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    print("=========================================================")
    print("STEP 1: Loading Dataset & Running Preprocessing Pipeline")
    print("=========================================================")
    patterns, tags, responses = load_intents_data(data_path)
    
    pipeline = IntentDataPipeline(max_sequence_length=16)
    X, y = pipeline.fit_transform(patterns, tags, responses)
    pipeline.save(save_dir)
    print(f"Total samples: {len(X)} | Vocab size: {pipeline.vocab_size} | Classes: {pipeline.num_classes}")

    # Stratified train/validation split (80% train, 20% validation)
    integer_labels = np.argmax(y, axis=1)
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=integer_labels,
    )
    print(f"Training split: {X_train.shape[0]} samples | Validation split: {X_val.shape[0]} samples")

    # Persist validation set for evaluation stage
    np.savez_compressed(save_path / "val_data.npz", X_val=X_val, y_val=y_val)

    # Common Callbacks
    def get_callbacks(model_name: str):
        return [
            EarlyStopping(
                monitor="val_loss",
                patience=15,
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=6,
                min_lr=1e-5,
                verbose=1,
            ),
            ModelCheckpoint(
                filepath=str(save_path / f"best_{model_name}.keras"),
                monitor="val_loss",
                save_best_only=True,
                verbose=0,
            ),
        ]

    # =========================================================
    # TRAIN 1: Standard LSTM Model
    # =========================================================
    print("\n=========================================================")
    print("STEP 2: Training Model A - Standard LSTM")
    print("=========================================================")
    lstm_model = build_lstm_model(
        vocab_size=pipeline.vocab_size,
        num_classes=pipeline.num_classes,
        max_sequence_length=pipeline.max_sequence_length,
    )
    
    start_t = time.time()
    lstm_history = lstm_model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=get_callbacks("lstm"),
        verbose=1,
    )
    lstm_duration = round(time.time() - start_t, 2)
    lstm_model.save(save_path / "lstm_model.keras")

    with open(save_path / "lstm_history.json", "w") as f:
        json.dump(lstm_history.history, f, indent=2)

    val_loss_lstm, val_acc_lstm = lstm_model.evaluate(X_val, y_val, verbose=0)
    print(f"-> Standard LSTM Finished in {lstm_duration}s | Val Loss: {val_loss_lstm:.4f} | Val Accuracy: {val_acc_lstm*100:.2f}%")

    # =========================================================
    # TRAIN 2: Bidirectional LSTM Model
    # =========================================================
    print("\n=========================================================")
    print("STEP 3: Training Model B - Bidirectional LSTM (Bi-LSTM)")
    print("=========================================================")
    bilstm_model = build_bilstm_model(
        vocab_size=pipeline.vocab_size,
        num_classes=pipeline.num_classes,
        max_sequence_length=pipeline.max_sequence_length,
    )

    start_t = time.time()
    bilstm_history = bilstm_model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=get_callbacks("bilstm"),
        verbose=1,
    )
    bilstm_duration = round(time.time() - start_t, 2)
    bilstm_model.save(save_path / "bilstm_model.keras")

    with open(save_path / "bilstm_history.json", "w") as f:
        json.dump(bilstm_history.history, f, indent=2)

    val_loss_bi, val_acc_bi = bilstm_model.evaluate(X_val, y_val, verbose=0)
    print(f"-> Bidirectional LSTM Finished in {bilstm_duration}s | Val Loss: {val_loss_bi:.4f} | Val Accuracy: {val_acc_bi*100:.2f}%")

    print("\n=========================================================")
    print("TRAINING COMPLETE! Both models and artifacts saved in 'saved_models/'.")
    print("=========================================================")

    return {
        "lstm_val_acc": val_acc_lstm,
        "lstm_val_loss": val_loss_lstm,
        "bilstm_val_acc": val_acc_bi,
        "bilstm_val_loss": val_loss_bi,
    }


if __name__ == "__main__":
    train_models()
