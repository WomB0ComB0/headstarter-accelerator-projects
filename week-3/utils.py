#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from functools import lru_cache
import os
from typing import Tuple, Any, Dict, Optional
import numpy as np
import cv2
import PIL.Image
import tensorflow as tf
from dataclasses import dataclass
import requests
import logging
from pathlib import Path
import google.generativeai as genai
import kagglehub
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ImageProcessingConfig:
    """Configuration for image processing parameters"""

    blur_kernel: Tuple[int, int] = (11, 11)
    threshold_percentile: float = 80
    heatmap_alpha: float = 0.7
    original_alpha: float = 0.3
    target_size: Tuple[int, int] = (299, 299)


class ModelRegistry:
    """Registry for managing different model configurations"""

    MODELS = {
        "xception": {
            "input_shape": (299, 299, 3),
            "weights_file": "xception_model.weights.h5",
            "url": "https://www.kaggle.com/api/v1/models/mikeodnis/brain_tumor_cnn/tensorFlow2/xception_model/1/download",
        },
        "cnn": {
            "input_shape": (224, 224, 3),
            "weights_file": "cnn_model.h5",
            "url": "https://www.kaggle.com/api/v1/models/mikeodnis/brain_tumosr_cnn/tensorFlow2/cnn_model/1/download",
        },
    }

    @staticmethod
    def get_model_config(model_type: str) -> Optional[Dict[str, Any]]:
        """Get model configuration by type"""
        return ModelRegistry.MODELS.get(model_type.lower())


def setup_model_directory(base_path: str = "models") -> str:
    """Setup directory for model storage"""
    model_dir = Path(base_path)
    model_dir.mkdir(parents=True, exist_ok=True)
    return str(model_dir)


@lru_cache(maxsize=2)
def load_model(model_type: str, weights_path: Optional[str] = None) -> tf.keras.Model:
    """Load and cache the specified model"""
    try:
        model_config = ModelRegistry.get_model_config(
            "xception" if "xception" in model_type.lower() else "cnn"
        )

        if model_config is None:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Download model using kagglehub
        logger.info("Downloading model from Kaggle...")
        model_name = "xception_model.weights.h5" if "xception" in model_type.lower() else "cnn_model.h5"
        model_path = kagglehub.model_download(
            "mikeodnis/brain_tumor_cnn/tensorFlow2/default"
        )
        weights_path = os.path.join(model_path, model_name)
        logger.info(f"Model downloaded to: {weights_path}")

        if "xception" in model_type.lower():
            return load_xception_model(weights_path)
        else:
            return load_cnn_model(weights_path)
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


def load_xception_model(weights_path: str) -> tf.keras.Model:
    """Load and configure Xception model"""
    try:
        img_shape = ModelRegistry.MODELS["xception"]["input_shape"]
        base_model = tf.keras.applications.Xception(
            input_shape=img_shape, include_top=False, weights="imagenet", pooling="max"
        )

        model = tf.keras.Sequential(
            [
                base_model,
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Dense(4, activation="softmax"),
            ]
        )

        model.build((None,) + img_shape)
        model.load_weights(weights_path)
        logger.info(f"Successfully loaded Xception model from {weights_path}")
        return model

    except Exception as e:
        logger.error(f"Error loading Xception model: {str(e)}")
        raise


def load_cnn_model(weights_path: str) -> tf.keras.Model:
    """Load and configure custom CNN model"""
    try:
        model = tf.keras.models.load_model(weights_path)
        logger.info(f"Successfully loaded CNN model from {weights_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading CNN model: {str(e)}")
        raise


def download_model_weights(model_type: str, base_path: str = "models") -> str:
    """Download model weights if not present"""
    model_config = ModelRegistry.get_model_config(model_type)
    if not model_config:
        raise ValueError(f"Invalid model type: {model_type}")

    model_dir = setup_model_directory(base_path)
    weights_path = os.path.join(model_dir, model_config["weights_file"])

    if not os.path.exists(weights_path):
        logger.info(f"Downloading weights for {model_type} model...")
        try:
            response = requests.get(model_config["url"], stream=True)
            response.raise_for_status()

            with open(weights_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"Successfully downloaded weights to {weights_path}")
        except Exception as e:
            logger.error(f"Error downloading model weights: {str(e)}")
            raise

    return weights_path


def preprocess_image(
    image: PIL.Image.Image, target_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """Preprocess image for model input"""
    if target_size:
        image = image.resize(target_size)

    img_array = np.array(image)
    img_array = img_array.astype(np.float32) / 255.0

    if len(img_array.shape) == 2:  # Convert grayscale to RGB
        img_array = np.stack((img_array,) * 3, axis=-1)

    return np.expand_dims(img_array, axis=0)


def create_circular_mask(img_size: Tuple[int, int]) -> np.ndarray:
    """Create a circular mask for the brain region"""
    center = (img_size[0] // 2, img_size[1] // 2)
    radius = min(center[0], center[1]) - 10
    y, x = np.ogrid[: img_size[0], : img_size[1]]
    return (pow((x - center[0]), 2) + pow((y - center[1]), 2)) <= pow(radius, 2)


def normalize_gradients(gradients: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Normalize gradients within the brain region"""
    brain_gradients = gradients[mask]
    if brain_gradients.max() > brain_gradients.min():
        brain_gradients = (brain_gradients - brain_gradients.min()) / (
            brain_gradients.max() - brain_gradients.min()
        )
    gradients[mask] = brain_gradients
    return gradients


def generate_saliency_map(
    model: tf.keras.Model,
    img_array: np.ndarray,
    class_index: int,
    config: ImageProcessingConfig = ImageProcessingConfig(),
) -> np.ndarray:
    """Generate saliency map for model visualization"""
    try:
        # Calculate gradients
        with tf.GradientTape() as tape:
            img_tensor = tf.convert_to_tensor(img_array)
            tape.watch(img_tensor)
            predictions = model(img_tensor)
            target_class = predictions[:, class_index]

        # Process gradients
        gradients = tf.abs(tape.gradient(target_class, img_tensor))
        gradients = tf.reduce_max(gradients, axis=-1).numpy().squeeze()
        gradients = cv2.resize(gradients, config.target_size)

        # Apply circular mask
        mask = create_circular_mask(config.target_size)
        gradients *= mask

        # Normalize and threshold gradients
        gradients = normalize_gradients(gradients, mask)
        threshold = np.percentile(gradients[mask], config.threshold_percentile)
        gradients[gradients < threshold] = 0

        # Apply Gaussian blur
        gradients = cv2.GaussianBlur(gradients, config.blur_kernel, 0)

        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * gradients), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Combine with original image
        original_img = cv2.resize(img_array[0] * 255, config.target_size)
        superimposed_img = (
            heatmap * config.heatmap_alpha + original_img * config.original_alpha
        )

        return superimposed_img.astype(np.uint8)

    except Exception as e:
        logger.error(f"Error generating saliency map: {str(e)}")
        raise


def save_results(
    original_img: np.ndarray, saliency_map: np.ndarray, output_dir: str, filename: str
) -> Tuple[str, str]:
    """Save original image and saliency map"""
    try:
        os.makedirs(output_dir, exist_ok=True)

        original_path = os.path.join(output_dir, f"original_{filename}")
        saliency_path = os.path.join(output_dir, f"saliency_{filename}")

        cv2.imwrite(original_path, cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(saliency_path, cv2.cvtColor(saliency_map, cv2.COLOR_RGB2BGR))

        logger.info(f"Saved results to {output_dir}")
        return original_path, saliency_path

    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        raise


def generate_explanation(
    img_path: str,
    model_prediction: str,
    confidence: float,
) -> str:
    """Generate explanation using Gemini with improved prompt"""
    try:
        # Get API key from environment or secrets
        if os.getenv("STREAMLIT_RUNTIME"):
            api_key = st.secrets["GOOGLE_API_KEY"]
        else:
            api_key = os.getenv("GOOGLE_API_KEY")
        
        if not api_key:
            raise ValueError("Google API key not found in environment variables or secrets")

        # Configure Gemini
        genai.configure(api_key=api_key)

        prompt = f"""You are an expert neurologist analyzing a brain tumor MRI scan saliency map.
        
        Context:
        - The deep learning model predicted: {model_prediction}
        - Confidence level: {confidence * 100:.1f}%
        - The light cyan regions indicate areas of model focus
        
        Please provide a 4-sentence analysis that:
        1. Identifies specific brain regions highlighted in the saliency map
        2. Explains the anatomical significance of these regions
        3. Connects these observations to the model's prediction
        4. Evaluates the confidence level in context of the visible features
        
        Focus on medical accuracy and clarity while avoiding technical ML terminology.
        """

        img = PIL.Image.open(img_path)
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        response = model.generate_content([prompt, img])

        return response.text
    except Exception as e:
        logger.error(f"Error in generate_explanation: {str(e)}")
        raise
