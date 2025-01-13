import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import typer
from loguru import logger
from keras._tf_keras.keras.models import load_model
from financial_market_crash_anomaly_detection_and_investment_strategy.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    PREDICTIONS_DIR,
)

app = typer.Typer()


def detect_anomalies(model, sequences, threshold=0.95):
    """
    Detect anomalies using the trained model.
    Returns anomaly scores and binary flags for anomalous periods.
    """
    # Get reconstruction error
    predictions = model.predict(sequences)

    # Flatten both arrays to calculate MSE
    original_shape = sequences.shape
    flattened_sequences = sequences.reshape(original_shape[0], -1)
    flattened_predictions = predictions.reshape(original_shape[0], -1)

    # Calculate MSE for each sequence
    mse = np.mean(np.power(flattened_sequences - flattened_predictions, 2), axis=1)

    # Calculate threshold based on prediction errors
    threshold_value = np.percentile(mse, threshold * 100)

    # Flag anomalies
    anomalies = mse > threshold_value

    return mse, anomalies


@app.command()
def main(
    model_path: Path = MODELS_DIR / "anomaly_detector.h5",
    scaler_path: Path = MODELS_DIR / "scaler.pkl",
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    output_path: Path = PREDICTIONS_DIR / "anomalies.csv",
    threshold: float = 0.95,
):
    # Load model and scaler
    logger.info("Loading model and scaler...")
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)

    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    features = pd.read_csv(features_path, index_col=0)

    # Scale features
    scaled_features = scaler.transform(features)

    # Create sequences
    seq_length = 30  # Must match training sequence length
    sequences = []
    for i in range(len(scaled_features) - seq_length):
        sequence = scaled_features[i : i + seq_length]
        sequences.append(sequence)
    sequences = np.array(sequences)

    # Detect anomalies
    logger.info("Detecting anomalies...")
    anomaly_scores, anomaly_flags = detect_anomalies(model, sequences, threshold)

    # Create results DataFrame
    results = pd.DataFrame(
        {"anomaly_score": anomaly_scores, "is_anomaly": anomaly_flags},
        index=features.index[seq_length:],
    )

    # Save results
    results.to_csv(output_path)
    logger.success(f"Anomaly detection complete! Results saved to {output_path}")

    # Print summary
    anomaly_periods = results[results["is_anomaly"]].index
    logger.info(f"Detected {len(anomaly_periods)} anomalous periods")
    if len(anomaly_periods) > 0:
        logger.info("Top 5 anomalous periods:")
        top_anomalies = results.nlargest(5, "anomaly_score")
        for idx, row in top_anomalies.iterrows():
            logger.info(f"{idx}: score = {row['anomaly_score']:.4f}")


if __name__ == "__main__":
    app()
