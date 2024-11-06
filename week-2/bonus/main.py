#! /usr/bin/env python3
# bonus/main.py

import streamlit as st
import pandas as pd
import pickle
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import time
from typing import Dict, Any, Optional
import logging
from sklearn.base import BaseEstimator
import utils as utils

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better UI with dark theme
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
        padding: 2rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .fraud-alert {
        background-color: rgba(255, 23, 68, 0.2);
        border-left: 5px solid #ff1744;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
        color: #ffffff;
    }
    .safe-alert {
        background-color: rgba(0, 200, 83, 0.2);
        border-left: 5px solid #00c853;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
        color: #ffffff;
    }
    .metric-container {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        color: #ffffff;
    }
    .metric-container p {
        color: #ffffff !important;
    }
    div[data-testid="stMetricValue"] {
        color: #ffffff;
    }
    div[data-testid="stMetricLabel"] {
        color: rgba(255, 255, 255, 0.8);
    }
    </style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def initialize_openai_client() -> Optional[OpenAI]:
    """Initialize OpenAI client with error handling."""
    try:
        return OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=st.secrets["GROQ_API_KEY"],
        )
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        st.error("⚠️ AI analysis features are currently unavailable.")
        return None


@st.cache_resource
def load_models(model_path: str) -> Dict[str, BaseEstimator]:
    """Load ML models with error handling and caching."""
    models = {}
    model_files = ["rf_model.pkl", "xgb_model.pkl", "dt_model.pkl"]

    for model_file in model_files:
        try:
            file_path = Path(model_path) / model_file
            with open(file_path, "rb") as file:
                model_name = model_file.split(".")[0]
                models[model_name] = pickle.load(file)
                logger.info(f"Successfully loaded {model_name}")
        except Exception as e:
            logger.error(f"Error loading {model_file}: {e}")

    if not models:
        st.error(
            "❌ Critical Error: No models could be loaded. Please check model files."
        )
        st.stop()

    return models


def create_transaction_summary(input_data: Dict[str, Any]) -> None:
    """Display transaction summary in a clean, organized manner."""
    st.markdown("### 📊 Transaction Summary")

    cols = st.columns(4)

    metrics = [
        ("💰 Amount", f"${input_data['amt']:.2f}"),
        ("👤 Customer Age", f"{input_data['age']} years"),
        ("📍 Distance", f"{input_data['distance']:.1f} miles"),
        ("🏢 City Pop.", f"{input_data['city_pop']:,}"),
        ("🏷️ Category", input_data["category"]),
        ("⚧ Gender", input_data["gender"]),
        ("💼 Occupation", input_data["job"]),
        ("🕒 Time", f"{input_data['hour']:02d}:00"),
    ]

    for i, (label, value) in enumerate(metrics):
        with cols[i % 4]:
            st.markdown(
                f"""
            <div class="metric-container">
                <p style='color: #666; margin-bottom: 0.2rem;'>{label}</p>
                <p style='font-size: 1.2em; font-weight: bold; margin: 0;'>{value}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )


def make_predictions(input_data: pd.DataFrame, models: Dict[str, BaseEstimator]) -> Dict[str, Dict[str, Any]]:
    """Make predictions with ensemble voting and confidence scores."""
    results = {}
    weights = {
        "rf_model": 0.4,
        "xgb_model": 0.3,
        "dt_model": 0.3,
    }
    
    # Preprocess input data
    processed_data = utils.prepare_features(input_data)
    
    ensemble_proba = 0
    total_weight = 0

    for name, model in models.items():
        try:
            prediction = model.predict(processed_data)[0]
            probability = model.predict_proba(processed_data)[0][1]

            weight = weights.get(name, 0.25)
            ensemble_proba += probability * weight
            total_weight += weight

            results[name] = {
                "prediction": "Fraudulent" if prediction == 1 else "Legitimate",
                "probability": probability,
            }
        except Exception as e:
            logger.error(f"Error with {name} model: {e}")
            st.warning(f"⚠️ {name.upper()} model prediction failed")

    if total_weight > 0:
        ensemble_proba /= total_weight
        results["ensemble"] = {
            "prediction": "Fraudulent" if ensemble_proba > 0.5 else "Legitimate",
            "probability": ensemble_proba,
        }

    return results


def create_visualization(results: Dict[str, Dict[str, Any]]) -> None:
    """Create interactive visualizations for the prediction results."""
    col1, col2 = st.columns(2)

    with col1:
        # Confidence gauge
        ensemble_prob = results.get("ensemble", {}).get("probability", 0)
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=ensemble_prob * 100,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Fraud Risk Score"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "darkred"},
                    "steps": [
                        {"range": [0, 30], "color": "lightgreen"},
                        {"range": [30, 70], "color": "lightyellow"},
                        {"range": [70, 100], "color": "salmon"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 70,
                    },
                },
            )
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col2:
        # Model comparison
        model_names = [k for k in results.keys() if k != "ensemble"]
        probabilities = [results[k]["probability"] for k in model_names]

        fig_comparison = px.bar(
            x=model_names,
            y=probabilities,
            title="Model Predictions Comparison",
            labels={"x": "Model", "y": "Fraud Probability"},
            color=probabilities,
            color_continuous_scale="RdYlGn_r",
        )
        fig_comparison.update_layout(showlegend=False)
        st.plotly_chart(fig_comparison, use_container_width=True)


def get_ai_analysis(
    client: OpenAI, input_data: Dict[str, Any], results: Dict[str, Dict[str, Any]]
) -> str:
    """Get AI analysis of the transaction with error handling and retries."""
    if not client:
        return "AI analysis unavailable"

    predictions_text = "\n".join(
        [
            f"- {name.upper()}: {result['prediction']} ({result['probability']:.2%} confidence)"
            for name, result in results.items()
        ]
    )

    prompt = f"""As a fraud detection expert, analyze this transaction:

Transaction Details:
- Amount: ${input_data['amt']:.2f}
- Category: {input_data['category']}
- Distance: {input_data['distance']} miles
- Hour: {input_data['hour']:02d}:00
- Customer Age: {input_data['age']}
- Occupation: {input_data['job']}

Model Predictions:
{predictions_text}

Provide a concise analysis including:
1. Risk assessment
2. Key suspicious/safe indicators
3. Recommended action

Format in markdown with clear sections."""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=200,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"AI analysis failed after {max_retries} attempts: {e}")
                return "Unable to generate AI analysis at this time."
            time.sleep(1)


def main():
    st.title("🛡️ Credit Card Fraud Detection System")
    st.markdown(
        """
    This advanced system uses machine learning to detect potential credit card fraud.
    Enter transaction details below for real-time analysis.
    """
    )

    # Initialize services
    client = initialize_openai_client()
    models = load_models("week-2/bonus/out/models")

    # Create input form
    with st.form("transaction_form"):
        col1, col2 = st.columns(2)

        with col1:
            amount = st.number_input("Transaction Amount ($)", min_value=0.0, step=0.01)
            age = st.number_input("Customer Age", min_value=18, max_value=100)
            distance = st.number_input("Distance from Home (miles)", min_value=0.0)
            city_pop = st.number_input("City Population", min_value=0)

        with col2:
            category = st.selectbox(
                "Transaction Category",
                [
                    "grocery_pos",
                    "shopping_pos",
                    "entertainment",
                    "food_dining",
                    "health_fitness",
                ],
            )
            gender = st.selectbox("Gender", ["M", "F"])
            job = st.selectbox(
                "Occupation",
                ["tech", "service", "professional", "student", "retired", "other"],
            )
            hour = st.number_input(
                "Hour of Transaction (0-23)", min_value=0, max_value=23
            )

        submitted = st.form_submit_button("🔍 Analyze Transaction")

    if submitted:
        with st.spinner("Analyzing transaction..."):
            # Prepare input data
            input_data = {
                "amt": amount,
                "age": age,
                "distance": distance,
                "city_pop": city_pop,
                "category": category,
                "gender": gender,
                "job": job,
                "hour": hour,
            }

            df_input = pd.DataFrame([input_data])

            try:
                # Create transaction summary
                create_transaction_summary(input_data)

                # Make predictions
                results = make_predictions(df_input, models)

                # Display results
                st.markdown("---")
                st.subheader("🔍 Detection Results")

                ensemble_result = results.get("ensemble", {})
                is_fraud = ensemble_result.get("probability", 0) > 0.7

                if is_fraud:
                    st.markdown(
                        """
                    <div class="fraud-alert">
                        <h3>⚠️ High Fraud Risk Detected</h3>
                        <p>This transaction shows strong indicators of fraudulent activity.</p>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        """
                    <div class="safe-alert">
                        <h3>✅ Transaction Appears Safe</h3>
                        <p>No significant fraud indicators detected.</p>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                # Create visualizations
                create_visualization(results)

                # Get and display AI analysis
                st.markdown("---")
                st.subheader("🤖 AI Analysis")
                analysis = get_ai_analysis(client, input_data, results)
                st.markdown(analysis)

                # Log the analysis
                logger.info(
                    f"Analysis completed - Transaction Amount: ${amount:.2f}, "
                    f"Fraud Probability: {ensemble_result.get('probability', 0):.2%}"
                )

            except Exception as e:
                logger.error(f"Error in analysis pipeline: {e}")
                st.error("An error occurred during analysis. Please try again.")

    # Add helpful information
    with st.expander("ℹ️ About the System"):
        st.markdown(
            """
        ### How it Works
        This system uses an ensemble of machine learning models:
        - 🌳 Random Forest (40% weight)
        - 🚀 XGBoost (30% weight)
        - 🌲 Decision Tree (30% weight)
        
        ### Interpreting Results
        - Risk Score > 70%: High risk of fraud
        - Risk Score 30-70%: Medium risk, requires review
        - Risk Score < 30%: Low risk
        
        ### Best Practices
        1. Always verify high-risk transactions
        2. Consider multiple factors, not just the risk score
        3. Monitor patterns over time
        """
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error("An unexpected error occurred. Please refresh the page and try again.")
