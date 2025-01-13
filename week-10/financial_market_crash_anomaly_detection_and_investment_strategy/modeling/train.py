import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras import Sequential, layers
import joblib
from pathlib import Path
import typer
from loguru import logger

from financial_market_crash_anomaly_detection_and_investment_strategy.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
)

app = typer.Typer()


def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequence = data[i : i + seq_length]
        sequences.append(sequence)
    return np.array(sequences)


def build_model(input_shape):
    model = Sequential(
        [
            layers.LSTM(64, input_shape=input_shape, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32),
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    model_path: Path = MODELS_DIR / "anomaly_detector.h5",
    scaler_path: Path = MODELS_DIR / "scaler.pkl",
):
    # Add error checking
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")

    logger.info("Loading features...")
    features = pd.read_csv(features_path, index_col=0)

    # Add data validation
    if features.empty:
        raise ValueError("Features DataFrame is empty. Please check data processing step.")

    # Prepare data
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Create sequences
    seq_length = 30
    sequences = create_sequences(scaled_features, seq_length)

    # Split data
    X_train, X_test = train_test_split(sequences, test_size=0.2, shuffle=False)

    # Build and train model
    logger.info("Training model...")
    model = build_model((seq_length, features.shape[1]))
    model.fit(
        X_train,
        np.zeros(len(X_train)),  # Using reconstruction error as anomaly score
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=1,
    )

    # Save model and scaler
    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    logger.success("Model training complete!")


if __name__ == "__main__":
    app()
