"""Neural Network architectures for E-commerce Intent Classification: LSTM vs Bi-LSTM."""

from typing import Tuple
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Embedding,
    SpatialDropout1D,
    LSTM,
    Bidirectional,
    Dense,
    Dropout,
)


def build_lstm_model(
    vocab_size: int,
    num_classes: int,
    max_sequence_length: int = 16,
    embedding_dim: int = 64,
    lstm_units: int = 64,
    dropout_rate: float = 0.3,
) -> tf.keras.Model:
    """Build a Standard Uni-directional LSTM classification model.
    
    Architecture:
        Input -> Embedding -> SpatialDropout1D -> LSTM -> Dense(ReLU) -> Dropout -> Dense(Softmax)
    """
    model = Sequential(
        [
            Input(shape=(max_sequence_length,), name="input_sequence"),
            Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                name="embedding_layer",
            ),
            SpatialDropout1D(0.25, name="spatial_dropout"),
            LSTM(
                units=lstm_units,
                return_sequences=False,
                dropout=0.2,
                name="lstm_layer",
            ),
            Dense(32, activation="relu", name="dense_intermediate"),
            Dropout(dropout_rate, name="dense_dropout"),
            Dense(num_classes, activation="softmax", name="output_intents"),
        ],
        name="Standard_LSTM_Classifier",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_bilstm_model(
    vocab_size: int,
    num_classes: int,
    max_sequence_length: int = 16,
    embedding_dim: int = 64,
    lstm_units: int = 64,
    dropout_rate: float = 0.3,
) -> tf.keras.Model:
    """Build a Bidirectional LSTM (Bi-LSTM) classification model.
    
    Architecture:
        Input -> Embedding -> SpatialDropout1D -> Bidirectional(LSTM) -> Dense(ReLU) -> Dropout -> Dense(Softmax)
    """
    model = Sequential(
        [
            Input(shape=(max_sequence_length,), name="input_sequence"),
            Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                name="embedding_layer",
            ),
            SpatialDropout1D(0.25, name="spatial_dropout"),
            Bidirectional(
                LSTM(
                    units=lstm_units,
                    return_sequences=False,
                    dropout=0.2,
                ),
                name="bidirectional_lstm",
            ),
            Dense(32, activation="relu", name="dense_intermediate"),
            Dropout(dropout_rate, name="dense_dropout"),
            Dense(num_classes, activation="softmax", name="output_intents"),
        ],
        name="Bidirectional_LSTM_Classifier",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
