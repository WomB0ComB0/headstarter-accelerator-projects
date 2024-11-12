#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from functools import lru_cache
import os
import sys
import streamlit as st
import kagglehub
import logging
from typing import Tuple, List, Set, Optional
import numpy as np
import plotly.graph_objects as go
import PIL.Image
from dotenv import load_dotenv

import utils

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR: str = os.path.join(BASE_DIR, "saliency_maps")
MODELS_DIR: str = os.path.join(BASE_DIR, "models")
IMG_FORMATS: Set[str] = {"jpg", "jpeg", "png"}
LABELS: List[str] = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]


@dataclass
class ModelConfig:
    path: str
    img_size: Tuple[int, int]
    name: str

    @classmethod
    def from_type(cls, model_type: str) -> "ModelConfig":
        """Create ModelConfig based on model type"""
        return cls(
            path=get_model_path(model_type), img_size=(224, 224), name=model_type
        )


class SystemCompatibilityCheck:
    """Check system compatibility"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def check_cpu_features(self) -> Tuple[bool, Optional[str]]:
        """Check CPU compatibility"""
        try:
            import cpuinfo

            info = cpuinfo.get_cpu_info()
            features = info.get("flags", [])
            required_features = {"avx", "sse4_1", "sse4_2"}
            missing_features = required_features - set(features)

            if missing_features:
                return (
                    False,
                    f"CPU missing required features: {', '.join(missing_features)}",
                )
            return True, None
        except Exception as e:
            self.logger.warning(f"Could not check CPU features: {str(e)}")
            return True, None


def initialize_app() -> None:
    """Initialize the application with compatibility checks"""
    checker = SystemCompatibilityCheck()

    cpu_compatible, cpu_error = checker.check_cpu_features()
    if not cpu_compatible:
        st.error(
            f"""
        System CPU Compatibility Issue Detected
        
        {cpu_error}
        
        Please try:
        1. Running on a newer CPU with AVX support
        2. Using a different hardware configuration
        3. Running with reduced model complexity
        """
        )
        sys.exit(1)


def setup_environment() -> None:
    """Initialize environment and create necessary directories"""
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(MODELS_DIR, exist_ok=True)

        env_file = os.path.join(BASE_DIR, ".env")
        if os.path.exists(env_file):
            logger.info(f"Loading environment from {env_file}")
            load_dotenv(env_file)

            try:
                home_dir = os.path.expanduser("~")
                streamlit_dir = os.path.join(home_dir, ".streamlit")
                secrets_file = os.path.join(streamlit_dir, "secrets.toml")

                if not os.path.exists(streamlit_dir):
                    logger.info(f"Creating Streamlit directory at {streamlit_dir}")
                    os.makedirs(streamlit_dir, exist_ok=True)

                if not os.path.exists(secrets_file):
                    logger.info(f"Creating secrets file at {secrets_file}")
                    with open(secrets_file, "w", encoding="utf-8") as f:
                        ngrok_token = os.getenv("NGROK_AUTH_TOKEN", "")
                        google_key = os.getenv("GOOGLE_API_KEY", "")

                        f.write(f'NGROK_AUTH_TOKEN = "{ngrok_token}"\n')
                        f.write(f'GOOGLE_API_KEY = "{google_key}"\n')

                    logger.info("Successfully created secrets.toml")
                else:
                    logger.info("Secrets file already exists")

            except Exception as e:
                logger.error(f"Error creating secrets file: {str(e)}")
                pass
        else:
            logger.warning(f".env file not found at {env_file}")

    except Exception as e:
        logger.error(f"Error in setup_environment: {str(e)}")
        raise


@lru_cache(maxsize=1)
def get_model_path(model_type: str) -> str:
    """Cache and return the model path based on environment"""
    if is_streamlit_cloud():
        base_path = "week-3/models"
    else:
        base_path = MODELS_DIR

    model_name = (
        "xception_model.onnx" if "xception" in model_type.lower() else "cnn_model.onnx"
    )
    local_path = os.path.join(base_path, model_name)

    if not os.path.exists(local_path):
        kaggle_path = kagglehub.model_download(
            "mikeodnis/brain_tumor_cnn/tensorFlow2/default"
        )
        import shutil

        source_path = os.path.join(kaggle_path, model_name)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        shutil.copy2(source_path, local_path)

    return local_path


def get_api_keys() -> Tuple[str, str]:
    """Get API keys from environment or Streamlit secrets"""
    ngrok_token = ""
    google_key = ""

    try:
        if is_streamlit_cloud():
            ngrok_token = st.secrets.get("NGROK_AUTH_TOKEN", "")
            google_key = st.secrets.get("GOOGLE_API_KEY", "")
        else:
            ngrok_token = os.getenv("NGROK_AUTH_TOKEN", "")
            google_key = os.getenv("GOOGLE_API_KEY", "")
    except Exception as e:
        logger.warning(f"Error getting API keys: {str(e)}")

    if not ngrok_token or not google_key:
        logger.warning("One or more API keys are missing")

    return ngrok_token, google_key


def is_streamlit_cloud() -> bool:
    """Check if running on Streamlit Cloud"""
    return os.getenv("STREAMLIT_RUNTIME_ENVIRONMENT") is not None


def setup_streamlit_page() -> None:
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="Brain Tumor Classification",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get help": "mailto:mike@mikeodnis.dev",
            "Report a bug": "mailto:mike@mikeodnis.dev",
            "About": "Brain Tumor Classification App",
        },
    )

    st.markdown(
        """
        <style>
        .main { background-color: #0e1117; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def create_prediction_visualization(
    predictions: np.ndarray, predicted_label: str
) -> go.Figure:
    """Create and return a plotly visualization for predictions"""
    sorted_indices = np.argsort(predictions)[::-1]
    sorted_labels = [LABELS[i] for i in sorted_indices]
    sorted_probabilities = predictions[sorted_indices]

    fig = go.Figure(
        go.Bar(
            x=sorted_probabilities,
            y=sorted_labels,
            orientation="h",
            marker_color=[
                "red" if label == predicted_label else "blue" for label in sorted_labels
            ],
        )
    )

    fig.update_layout(
        title="Probabilities for each class",
        xaxis_title="Probability",
        yaxis_title="Class",
        height=400,
        width=600,
        yaxis=dict(autorange="reversed"),
    )

    for i, prob in enumerate(sorted_probabilities):
        fig.add_annotation(
            x=prob, y=i, text=f"{prob:.4f}", showarrow=False, xanchor="left", xshift=5
        )

    return fig


def display_results(predicted_label: str, confidence: float) -> None:
    """Display classification results in a styled container"""
    st.markdown(
        f"""
        <div style="background-color: #000000; color: #ffffff; padding: 30px; border-radius: 15px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="flex: 1; text-align: center;">
                    <h3 style="color: #ffffff; margin-bottom: 10px; font-size: 20px;">Prediction</h3>
                    <p style="font-size: 36px; font-weight: 800; color: #FF0000; margin: 0;">
                        {predicted_label}
                    </p>
                </div>
                <div style="width: 2px; height: 80px; background-color: #ffffff; margin: 0 20px;"></div>
                <div style="flex: 1; text-align: center;">
                    <h3 style="color: #ffffff; margin-bottom: 10px; font-size: 20px;">Confidence</h3>
                    <p style="font-size: 36px; font-weight: 800; color: #2196F3; margin: 0;">
                        {confidence:.4%}
                    </p>
                </div>
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Main application logic"""
    try:
        initialize_app()
        setup_environment()
        setup_streamlit_page()

        st.title("Brain Tumor Classification")
        st.write(
            "Upload an image of a brain MRI scan to classify if there is a tumor in the image."
        )

        upload_file = st.file_uploader("Choose an image...", type=list(IMG_FORMATS))

        if upload_file:
            with st.spinner("Processing image..."):
                # Model selection
                model_type = st.radio(
                    "Select a model:", ("Transfer Learning - Xception", "Custom CNN")
                )

                try:
                    model, input_shape = utils.load_model(
                        model_type,
                        weights_path=os.path.join(
                            MODELS_DIR,
                            (
                                "xception_model.onnx"
                                if "xception" in model_type.lower()
                                else "cnn_model.onnx"
                            ),
                        ),
                    )

                    model_config = ModelConfig(
                        path=get_model_path(model_type),
                        img_size=input_shape,
                        name=model_type,
                    )

                    img = PIL.Image.open(upload_file)
                    img_array = utils.preprocess_image(img, input_shape)

                    with st.spinner("Analyzing image..."):
                        input_name = model.get_inputs()[0].name
                        output_name = model.get_outputs()[0].name

                        predictions = model.run(
                            [output_name], {input_name: img_array.astype(np.float32)}
                        )[0]
                        predicted_class = np.argmax(predictions[0])
                        predicted_label = LABELS[predicted_class]

                except Exception as e:
                    st.error(f"Error in model prediction: {str(e)}")
                    logger.error(f"Model prediction error: {str(e)}")
                    return

                try:
                    processing_config = utils.ImageProcessingConfig(
                        target_size=model_config.img_size
                    )

                    with st.spinner("Generating visualization..."):
                        saliency_map = utils.generate_saliency_map(
                            model, img_array, predicted_class, processing_config
                        )

                        original_path, saliency_path = utils.save_results(
                            img_array[0] * 255,
                            saliency_map,
                            OUTPUT_DIR,
                            upload_file.name,
                        )

                except Exception as e:
                    st.error(f"Error generating visualization: {str(e)}")
                    logger.error(f"Visualization error: {str(e)}")
                    return

                col1, col2 = st.columns(2)
                with col1:
                    st.image(
                        upload_file, caption="Uploaded Image", use_column_width=True
                    )
                with col2:
                    st.image(
                        saliency_map, caption="Saliency Map", use_column_width=True
                    )

                st.write("## Classification Results")
                display_results(predicted_label, predictions[0][predicted_class])

                with st.spinner("Creating visualization..."):
                    fig = create_prediction_visualization(
                        predictions[0], predicted_label
                    )
                    st.plotly_chart(fig, use_container_width=True)

                if st.checkbox("Show AI Analysis", value=True):
                    try:
                        with st.spinner("Generating AI analysis..."):
                            explanation = utils.generate_explanation(
                                saliency_path,
                                predicted_label,
                                predictions[0][predicted_class],
                            )
                            st.write("## AI Analysis")
                            st.write(explanation)
                    except Exception as e:
                        st.warning("AI analysis not available at this time.")
                        logger.error(f"Explanation generation error: {str(e)}")

                col1, col2 = st.columns(2)
                with col1:
                    with open(original_path, "rb") as file:
                        st.download_button(
                            "Download Original Image",
                            file,
                            file_name=f"original_{upload_file.name}",
                            mime="image/png",
                        )
                with col2:
                    with open(saliency_path, "rb") as file:
                        st.download_button(
                            "Download Saliency Map",
                            file,
                            file_name=f"saliency_{upload_file.name}",
                            mime="image/png",
                        )

    except Exception as e:
        logger.exception("An unexpected error occurred in the main application")
        st.error(
            f"""
            An unexpected error occurred. Please try again or contact support.
            Error: {str(e)}
        """
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("An error occurred in the main application")
        st.error(f"An error occurred: {str(e)}")
