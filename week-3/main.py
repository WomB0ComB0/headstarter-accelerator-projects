#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from functools import lru_cache
import os
import streamlit as st
import kagglehub
from concurrent.futures import ThreadPoolExecutor
from pyngrok import ngrok
import logging
from typing import Tuple, List
import numpy as np
import plotly.graph_objects as go
import PIL.Image
from dotenv import load_dotenv

# Configure logging once at module level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
OUTPUT_DIR = 'saliency_maps'
IMG_FORMATS = {"jpg", "jpeg", "png"}
LABELS = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

@dataclass
class ModelConfig:
    path: str
    img_size: Tuple[int, int]
    name: str

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
    return os.getenv("NGROK_AUTH_TOKEN"), os.getenv("GOOGLE_API_KEY")

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

def create_prediction_visualization(predictions: np.ndarray, predicted_label: str) -> go.Figure:
    """Create and return a plotly visualization for predictions"""
    sorted_indices = np.argsort(predictions)[::-1]
    sorted_labels = [LABELS[i] for i in sorted_indices]
    sorted_probabilities = predictions[sorted_indices]
    
    fig = go.Figure(go.Bar(
        x=sorted_probabilities,
        y=sorted_labels,
        orientation='h',
        marker_color=['red' if label == predicted_label else 'blue' for label in sorted_labels]
    ))
    
    fig.update_layout(
        title='Probabilities for each class',
        xaxis_title='Probability',
        yaxis_title='Class',
        height=400,
        width=600,
        yaxis=dict(autorange='reversed')
    )
    
    for i, prob in enumerate(sorted_probabilities):
        fig.add_annotation(
            x=prob,
            y=i,
            text=f'{prob:.4f}',
            showarrow=False,
            xanchor='left',
            xshift=5
        )
    
    return fig

def display_results(predicted_label: str, confidence: float) -> None:
    """Display classification results in a styled container"""
    st.markdown(f"""
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
    """, unsafe_allow_html=True)

def main() -> None:
    """Main application logic"""
    setup_environment()
    setup_streamlit_page()
    
    st.title("Brain Tumor Classification")
    st.write("Upload an image of a brain MRI scan to classify if there is a tumor in the image.")
    
    upload_file = st.file_uploader("Choose an image...", type=list(IMG_FORMATS))
    
    if upload_file:
        model_type = st.radio(
            "Select a model:",
            ("Transfer Learning - Xception", "Custom CNN")
        )
        
        model_config = ModelConfig(
            path=get_model_path(),
            img_size=(299, 299) if model_type == "Transfer Learning - Xception" else (224, 224),
            name=model_type
        )
        
        # Process image and get predictions
        img = PIL.Image.open(upload_file).resize(model_config.img_size)
        img_array = np.array(img)[np.newaxis, ...] / 255.0
        
        # Load model and get predictions (implement these in utils)
        import utils
        model = (utils.load_xception_model if model_type == "Transfer Learning - Xception" 
                else utils.load_model)(f"{model_config.path}/model.h5")
        
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        predicted_label = LABELS[predicted_class]
        
        # Generate and display saliency map
        saliency_map = utils.generate_saliency_map(
            model, img_array, predicted_class, model_config.img_size, 
            img, upload_file, OUTPUT_DIR
        )
        
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.image(upload_file, caption='Uploaded Image', use_column_width=True)
        with col2:
            st.image(saliency_map, caption='Saliency Map', use_column_width=True)
            
        st.write("## Classification Results")
        display_results(predicted_label, predictions[0][predicted_class])
        
        # Show prediction visualization
        fig = create_prediction_visualization(predictions[0], predicted_label)
        st.plotly_chart(fig, use_container_width=True)
        
        # Generate and display explanation
        saliency_map_path = f'{OUTPUT_DIR}/{upload_file.name}'
        explanation = utils.generate_explanation(
            saliency_map_path, predicted_label, predictions[0][predicted_class]
        )
        st.write("## Explanation")
        st.write(explanation)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("An error occurred in the main application")
        st.error(f"An error occurred: {str(e)}")