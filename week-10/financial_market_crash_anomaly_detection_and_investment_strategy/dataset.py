import pandas as pd
import numpy as np
from pathlib import Path
import typer
from loguru import logger
from tqdm import tqdm

from financial_market_crash_anomaly_detection_and_investment_strategy.config import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
)

app = typer.Typer()


def clean_data(df):
    # Skip the metadata rows and get actual data
    data_start = df.index[df.index.str.contains("Date", na=False)].tolist()[0]
    df = df.loc[data_start:].reset_index(drop=True)

    # Set column names from the first row
    df.columns = df.iloc[0]
    df = df.iloc[1:]

    # Convert to numeric, dropping any non-numeric columns
    numeric_columns = []
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
            numeric_columns.append(col)
        except:
            continue

    df = df[numeric_columns]

    # Convert date column to datetime
    df.index = pd.to_datetime(df.index)

    # Remove rows with too many missing values
    df = df.dropna(thresh=len(df.columns) * 0.5)

    # Forward fill remaining missing values
    df = df.fillna(method="ffill")

    return df


def engineer_features(df):
    feature_dfs = []  # List to store individual feature DataFrames

    for col in df.columns:
        # Create features for this column
        col_features = pd.DataFrame(index=df.index)

        # Calculate returns and replace inf values with NaN
        returns = df[col].pct_change()
        returns = returns.replace([np.inf, -np.inf], np.nan)
        col_features[f"returns_{col}"] = returns

        # Calculate volatility
        volatility = df[col].rolling(window=20).std()
        col_features[f"volatility_{col}"] = volatility

        # Calculate moving averages
        ma50 = df[col].rolling(window=50).mean()
        ma200 = df[col].rolling(window=200).mean()

        # Calculate moving average crosses
        col_features[f"ma50_cross_{col}"] = (df[col] > ma50).astype(float)
        col_features[f"ma200_cross_{col}"] = (df[col] > ma200).astype(float)

        feature_dfs.append(col_features)

    # Combine all features using concat
    features = pd.concat(feature_dfs, axis=1)

    # Drop NaN values that result from rolling calculations
    features = features.dropna()

    # Replace any remaining inf values with NaN and drop those rows
    features = features.replace([np.inf, -np.inf], np.nan).dropna()

    return features


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "data.csv",
    interim_path: Path = INTERIM_DATA_DIR / "cleaned_data.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
):
    logger.info("Loading and processing dataset...")

    # Load raw data
    df = pd.read_csv(input_path, index_col=0)

    # Clean data
    logger.info("Cleaning data...")
    cleaned_data = clean_data(df)
    cleaned_data.to_csv(interim_path)

    # Engineer features
    logger.info("Engineering features...")
    features = engineer_features(cleaned_data)

    # Final check for inf values before saving
    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    features.to_csv(output_path)

    logger.success("Data processing complete!")


if __name__ == "__main__":
    app()
