#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from functools import lru_cache
import os
import sys
import streamlit as st
import kagglehub
from concurrent.futures import ThreadPoolExecutor
from pyngrok import ngrok
import logging
from typing import Tuple, List, Set, Dict, Optional
import numpy as np
import plotly.graph_objects as go
import PIL.Image
from dotenv import load_dotenv

# Import the utils module
import utils

# Configure logging once at module level
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
OUTPUT_DIR: str = "saliency_maps"
IMG_FORMATS: Set[str] = {"jpg", "jpeg", "png"}
LABELS: List[str] = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]


@dataclass
class ModelConfig:
    path: str
    img_size: Tuple[int, int]
    name: str


class SystemCompatibilityCheck:
    """Check system compatibility and handle TensorFlow/hardware issues"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def check_cpu_features(self) -> Tuple[bool, Optional[str]]:
        """Check CPU compatibility with TensorFlow"""
        try:
            cpu_info = platform.processor()
            # Check for minimum CPU features required by TensorFlow
            import cpuinfo

            features = cpuinfo.get_cpu_info().get("flags", [])
            required_features = {"avx", "sse4_1", "sse4_2"}
            missing_features = required_features - set(features)

            if missing_features:
                return (
                    False,
                    f"CPU missing required features: {', '.join(missing_features)}",
                )
            return True, None
        except Exception as e:
            return False, f"Error checking CPU features: {str(e)}"

    def check_tensorflow_compatibility(self) -> Tuple[bool, Optional[str]]:
        """Verify TensorFlow compatibility with system"""
        try:
            # Test basic TensorFlow operations
            tf.constant([1.0, 2.0])
            return True, None
        except Exception as e:
            return False, f"TensorFlow compatibility issue: {str(e)}"

    def setup_tensorflow_cpu(self) -> None:
        """Configure TensorFlow for CPU-only operation"""
        try:
            tf.config.set_visible_devices([], "GPU")
            tf.config.threading.set_intra_op_parallelism_threads(2)
            tf.config.threading.set_inter_op_parallelism_threads(2)
        except Exception as e:
            self.logger.warning(f"Error configuring TensorFlow: {str(e)}")


def initialize_app() -> None:
    """Initialize the application with compatibility checks"""
    checker = SystemCompatibilityCheck()

    # Check CPU compatibility
    cpu_compatible, cpu_error = checker.check_cpu_features()
    if not cpu_compatible:
        st.error(
            f"""
        System CPU Compatibility Issue Detected
        
        {cpu_error}
        
        Please try:
        1. Running on a newer CPU with AVX support
        2. Using a CPU-optimized version of TensorFlow
        3. Running with reduced model complexity
        """
        )
        sys.exit(1)

    # Check TensorFlow compatibility
    tf_compatible, tf_error = checker.check_tensorflow_compatibility()
    if not tf_compatible:
        st.error(
            f"""
        TensorFlow Compatibility Issue Detected
        
        {tf_error}
        
        Please try:
        1. Installing TensorFlow version compatible with your CPU
        2. Running with CPU-only configuration
        3. Checking system requirements
        """
        )
        sys.exit(1)

    # Configure TensorFlow for CPU operation
    checker.setup_tensorflow_cpu()


def setup_environment() -> None:
    """Initialize environment and create necessary directories"""
    load_dotenv()
    os.makedirs(OUTPUT_DIR, exist_ok=True)


@lru_cache(maxsize=1)
def get_model_path() -> str:
    """Cache and return the model path"""
    return kagglehub.model_download("mikeodnis/brain_tumor_cnn/tensorFlow2/default")


def get_api_keys() -> Tuple[str, str]:
    """Get API keys from environment or Streamlit secrets"""
    if is_streamlit_cloud():
        return st.secrets["NGROK_AUTH_TOKEN"], st.secrets["GOOGLE_API_KEY"]
    return os.getenv("NGROK_AUTH_TOKEN", ""), os.getenv("GOOGLE_API_KEY", "")


def is_streamlit_cloud() -> bool:
    """Check if running on Streamlit Cloud"""
    try:
        _ = st.secrets["NGROK_AUTH_TOKEN"]
        return True
    except KeyError:
        return False


def setup_streamlit_page() -> None:
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="Brain Tumor Classification",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={"Contact": "mailto:mike.odnis@mikeodnis.com"},
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
        setup_environment()
        setup_streamlit_page()

        st.title("Brain Tumor Classification")
        st.write(
            "Upload an image of a brain MRI scan to classify if there is a tumor in the image."
        )

        # File upload with progress and validation
        upload_file = st.file_uploader("Choose an image...", type=list(IMG_FORMATS))

        if upload_file:
            with st.spinner("Processing image..."):
                # Model selection
                model_type = st.radio(
                    "Select a model:", ("Transfer Learning - Xception", "Custom CNN")
                )

                # Configure model settings
                model_config = ModelConfig(
                    path=get_model_path(),
                    img_size=(
                        (299, 299)
                        if model_type == "Transfer Learning - Xception"
                        else (224, 224)
                    ),
                    name=model_type,
                )

                # Process image
                try:
                    img = PIL.Image.open(upload_file)
                    img_array = utils.preprocess_image(img, model_config.img_size)
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
                    logger.error(f"Image processing error: {str(e)}")
                    return

                # Load model and make predictions
                try:
                    model = utils.load_model(
                        model_type, weights_path=f"{model_config.path}/model.h5"
                    )

                    # Add progress bar for prediction
                    with st.spinner("Analyzing image..."):
                        predictions = model.predict(img_array)
                        predicted_class = np.argmax(predictions[0])
                        predicted_label = LABELS[predicted_class]

                except Exception as e:
                    st.error(f"Error in model prediction: {str(e)}")
                    logger.error(f"Model prediction error: {str(e)}")
                    return

                # Generate saliency map
                try:
                    # Create processing config
                    processing_config = utils.ImageProcessingConfig(
                        target_size=model_config.img_size
                    )

                    # Generate saliency map
                    with st.spinner("Generating visualization..."):
                        saliency_map = utils.generate_saliency_map(
                            model, img_array, predicted_class, processing_config
                        )

                        # Save results
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

                # Display results in columns
                col1, col2 = st.columns(2)
                with col1:
                    st.image(
                        upload_file, caption="Uploaded Image", use_column_width=True
                    )
                with col2:
                    st.image(
                        saliency_map, caption="Saliency Map", use_column_width=True
                    )

                # Display classification results
                st.write("## Classification Results")
                display_results(predicted_label, predictions[0][predicted_class])

                # Show prediction visualization
                with st.spinner("Creating visualization..."):
                    fig = create_prediction_visualization(
                        predictions[0], predicted_label
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Optional: Generate and display AI explanation
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

                # Add download buttons for results
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
