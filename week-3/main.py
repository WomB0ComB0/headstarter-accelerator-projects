#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import streamlit as st
import kagglehub
from threading import Thread
from pyngrok import ngrok
import time
import logging
import utils as utils
import tensorflow as tf
import tensorflow.keras.preprocessing.image as image
import numpy as np
import plotly.graph_objects as go
import cv2
from typing import Tuple, List, Dict, Set, Any, Union, Optional
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.metrics import Precision, Recall
import google.generative as genai
import PIL.Image
from dotenv import load_dotenv

load_dotenv(path=".env")

output_dir: str = 'saliency_maps'
os.makedirs(output_dir, exist_ok=True)

path: str = kagglehub.model_download("mikeodnis/brain_tumor_cnn/tensorFlow2/default")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger: logging.Logger = logging.getLogger(__name__)

def is_streamlit_cloud() -> bool:
    """Check if the application is running on Streamlit Cloud by attempting to access a secret"""
    try:
        _ = st.secrets["NGROK_AUTH_TOKEN"]
        return True
    except:
        return False

# Replace the API key assignments with:
if is_streamlit_cloud():
    ngrok_api_key = st.secrets["NGROK_AUTH_TOKEN"]
    gemini_api_key = st.secrets["GOOGLE_API_KEY"]
else:
    ngrok_api_key = os.getenv("NGROK_AUTH_TOKEN")
    gemini_api_key = os.getenv("GOOGLE_API_KEY")

st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Contact": "mailto:mike.odnis@mikeodnis.com",
    },
)

# Custom CSS
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Brain Tumor Classification")
st.write(
    "Upload an image of a brain MRI scan to classify if there is a tumor in the image."
)


def run_streamlit() -> None:
    os.system("streamlit run main.py --server.port 8501")


thread: Thread = Thread(target=run_streamlit)
thread.start()
time.sleep(5)
public_url: str = ngrok.connect(addr=8501, proto="http", bind_tls=True)

upload_file: st.uploaded_file_manager.UploadedFile = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"]
)

if upload_file is not None:
    select_model = st.radio(
        "Select a model:",
        ("Transfer Learning - Xception", "Custom CNN"),
    )

    if select_model == "Transfer Learning - Xception":
        model = utils.load_xception_model(f"{path}/model.h5")
        img_size: Tuple[int, int] = (299, 299)
    else:
        model = utils.load_model(f"{path}/model.h5")
        img_size: Tuple[int, int] = (224, 224)

    labels: List[str] = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
    img: PIL.Image.Image = image.load_img(upload_file, target_size=img_size)
    img_array: np.ndarray = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions: np.ndarray = model.predict(img_array)

    predicted_class: int = np.argmax(predictions[0])
    predicted_label: str = labels[predicted_class]

    st.write(f"Prediction: {predicted_label}")
    st.write("Predictions:")
    for label, prob in zip(labels, predictions[0]):
        st.write(f"{label}: {prob:.4f}")
        
    saliency_map: str = utils.generate_saliency_map(model, img_array, predicted_class, img_size, img, upload_file, output_dir)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(upload_file, caption='Uploaded Image', use_column_width=True)
    with col2:
        st.image(saliency_map, caption='Saliency Map', use_column_width=True)

    st.write("## Classification Results")
    
    result_container = st.container()
    result_container = st.container()
    result_container.markdown(f"""
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
                        {predictions[0][predicted_class]:.4%}
                    </p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    probabilities = predictions[0]
    sorted_indices = np.argsort(probabilities)[::-1]
    sorted_tables = [labels[i] for i in sorted_indices]
    sorted_probabilities = probabilities[sorted_indices]
    
    fig = go.Figure(go.Bar(
        x=sorted_probabilities, 
        y=sorted_tables,
        orientation='h',
        marker_color=['red' if label == predicted_label else 'blue' for label in sorted_tables]
    ))
    
    fig.update_layout(
        title='Probabilities for each class',
        xaxis_title ='Probability',
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
    
    st.plotly_chart(fig, use_container_width=True)
    
    saliency_map_path = f'saliency_maps/{upload_file.name}'
    explanation = utils.generate_explanation(saliency_map_path, predicted_label, predictions[0][predicted_class])
    
    st.write("## Explanation")
    st.write(explanation)
    
tunnels: List[ngrok.NgrokTunnel] = ngrok.get_tunnels()
for tunnel in tunnels:
    logger.info(
        f"Closing tunnel: {tunnel.public_url} -> {tunnel.config['addr']}"
    )
    ngrok.disconnect(tunnel.public_url)


def main() -> None:
    logger.info("Starting main function")


if __name__ == "__main__":
    print("Path to model files:", path)
    main()
