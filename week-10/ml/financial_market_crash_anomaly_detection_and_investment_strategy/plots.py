import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path
import typer
from loguru import logger
from tqdm import tqdm
from openai import OpenAI
import os
from financial_market_crash_anomaly_detection_and_investment_strategy.config import (
    FIGURES_DIR,
    PROCESSED_DATA_DIR,
)

app = typer.Typer()


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "features.csv",
    output_path: Path = FIGURES_DIR,
):
    logger.info("Generating plots from data...")

    # Load the processed data
    df = pd.read_csv(input_path)

    # Extract relevant columns
    feature_columns = [
        "returns_278.250",
        "volatility_278.250",
        "ma50_cross_278.250",
        "ma200_cross_278.250",
        "returns_1121.000",
        "volatility_1121.000",
        "ma50_cross_1121.000",
        "ma200_cross_1121.000",
        "returns_136.780",
        "volatility_136.780",
        "ma50_cross_136.780",
        "ma200_cross_136.780",
        "returns_100.240",
        "volatility_100.240",
        "ma50_cross_100.240",
        "ma200_cross_100.240",
        "returns_121.070",
        "volatility_121.070",
        "ma50_cross_121.070",
    ]

    # Ensure all columns exist in the dataframe
    df = df[feature_columns]

    # Example: Plotting the first feature column
    plt.figure(figsize=(10, 6))
    for column in feature_columns:
        plt.plot(df.index, df[column], label=column)
    plt.title("Feature Plot")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig(output_path / "feature_plot.png")
    plt.close()

    logger.success(f"Plot saved to {output_path}")

    # Use the GROQ API
    groq_api = os.getenv("GROQ_API_KEY")
    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=groq_api)

    query_data = df.head(5)
    system_prompt = """You are an expert financial analyst with machine learning background. You also have knowledge of how to detect anomalies in financial data.
    You will explain all column headers and their influence over the financial market.
    Explain their contributions to market anomalies.
    Out of all columns headers, explain which are the key columns required to train an ML model that can detect market anomalies.
    Explain if any columns need to be dropped in order to create an accurate model.
    """
    user_query = f"This is the data: {query_data}. variable Y in the dataframe is a binary indicator of a market anomaly or not."

    llm_response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
    )

    print(llm_response.choices[0].message.content)


if __name__ == "__main__":
    app()
