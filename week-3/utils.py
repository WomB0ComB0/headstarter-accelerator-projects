#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from functools import lru_cache
import os
from typing import Tuple, Any, Dict, Optional
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from dataclasses import dataclass
import requests
import logging
from pathlib import Path
import google.generativeai as genai
import kagglehub
import streamlit as st
import onnxruntime as ort

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ImageProcessingConfig:
    """Configuration for image processing parameters"""

    blur_radius: int = 2
    threshold_percentile: float = 80
    heatmap_alpha: float = 0.7
    original_alpha: float = 0.3
    target_size: Tuple[int, int] = (299, 299)


class ModelRegistry:
    """Registry for managing different model configurations"""

    MODELS = {
        "xception": {
            "input_shape": (299, 299, 3),
            "weights_file": "xception_model.onnx",
        },
        "cnn": {
            "input_shape": (224, 224, 3),
            "weights_file": "cnn_model.onnx",
        },
    }

    @staticmethod
    def get_model_config(model_type: str) -> Optional[Dict[str, Any]]:
        """Get model configuration by type"""
        key = "xception" if "xception" in model_type.lower() else "cnn"
        return ModelRegistry.MODELS.get(key)


def setup_model_directory(base_path: str = "models") -> str:
    """Setup directory for model storage"""
    model_dir = Path(base_path)
    model_dir.mkdir(parents=True, exist_ok=True)
    return str(model_dir)


@lru_cache(maxsize=2)
def load_model(model_type: str, weights_path: Optional[str] = None) -> Tuple[ort.InferenceSession, Tuple[int, int]]:
    """Load and cache the specified ONNX model"""
    try:
        model_config = ModelRegistry.get_model_config(
            "xception" if "xception" in model_type.lower() else "cnn"
        )

        if model_config is None:
            raise ValueError(f"Unsupported model type: {model_type}")

        if not weights_path:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(base_dir, "models")
            weights_path = os.path.join(models_dir, model_config["weights_file"])

        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model file not found at: {weights_path}")

        logger.info(f"Loading model from: {weights_path}")
        session = ort.InferenceSession(weights_path)
        input_shape = model_config["input_shape"][:2]
        logger.info(f"Successfully loaded ONNX model with input shape {input_shape}")
        return session, input_shape

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
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


def preprocess_image(image: Image.Image, target_size: Tuple[int, int]) -> np.ndarray:
    """Preprocess image for model input"""
    image = image.convert("RGB").resize(target_size, Image.Resampling.LANCZOS)

    img_array = np.array(image).astype(np.float32) / 255.0

    return np.expand_dims(img_array, axis=0)


def create_heatmap(gradients: np.ndarray, config: ImageProcessingConfig) -> Image.Image:
    """Create a heatmap visualization"""
    normalized = (
        (gradients - gradients.min()) * 255 / (gradients.max() - gradients.min())
    ).astype(np.uint8)

    heatmap = Image.fromarray(normalized).convert("L")

    heatmap = heatmap.filter(ImageFilter.GaussianBlur(radius=config.blur_radius))

    heatmap = ImageOps.colorize(heatmap, black="blue", mid="yellow", white="red")

    return heatmap


def blend_images(
    original: Image.Image, heatmap: Image.Image, config: ImageProcessingConfig
) -> Image.Image:
    """Blend original image with heatmap"""
    return Image.blend(
        original.convert("RGBA"), heatmap.convert("RGBA"), config.heatmap_alpha
    )


def save_results(
    original_img: Image.Image, heatmap: Image.Image, output_dir: str, filename: str
) -> Tuple[str, str]:
    """Save original image and heatmap"""
    try:
        os.makedirs(output_dir, exist_ok=True)

        original_path = os.path.join(output_dir, f"original_{filename}")
        heatmap_path = os.path.join(output_dir, f"heatmap_{filename}")

        original_img.save(original_path)
        heatmap.save(heatmap_path)

        logger.info(f"Saved results to {output_dir}")
        return original_path, heatmap_path

    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        raise


def generate_explanation(
    img_path: str,
    model_prediction: str,
    confidence: float,
) -> str:
    """Generate explanation using Gemini"""
    try:
        if os.getenv("STREAMLIT_RUNTIME"):
            api_key = st.secrets["GOOGLE_API_KEY"]
        else:
            api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError("Google API key not found")

        genai.configure(api_key=api_key)

        prompt = f"""You are an expert neurologist analyzing a brain MRI scan.
        
        Context:
        - The model predicted: {model_prediction}
        - Confidence level: {confidence * 100:.1f}%
        
        Please provide a 4-sentence analysis that:
        1. Identifies key features in the image
        2. Explains their anatomical significance
        3. Connects these observations to the prediction
        4. Evaluates the confidence level
        
        Focus on medical accuracy and clarity.
        """

        img = Image.open(img_path)
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        response = model.generate_content([prompt, img])

        return response.text
    except Exception as e:
        logger.error(f"Error in generate_explanation: {str(e)}")
        raise


def generate_saliency_map(
    model: ort.InferenceSession, 
    img_array: np.ndarray, 
    predicted_class: int,
    config: ImageProcessingConfig
) -> Image.Image:
    """Generate saliency map for the given image and model prediction"""
    try:
        input_name = model.get_inputs()[0].name
        output_name = model.get_outputs()[0].name
        
        img_tensor = img_array.copy()
        
        epsilon = 1e-5
        gradients = np.zeros_like(img_tensor)
        
        for i in range(3):
            for x in range(img_tensor.shape[1]):
                for y in range(img_tensor.shape[2]):
                    img_tensor[0, x, y, i] += epsilon
                    output_plus = model.run(
                        [output_name], 
                        {input_name: img_tensor.astype(np.float32)}
                    )[0][0][predicted_class]
                    
                    img_tensor[0, x, y, i] -= 2 * epsilon
                    output_minus = model.run(
                        [output_name], 
                        {input_name: img_tensor.astype(np.float32)}
                    )[0][0][predicted_class]
                    
                    img_tensor[0, x, y, i] += epsilon
                    
                    gradients[0, x, y, i] = (output_plus - output_minus) / (2 * epsilon)
        
        saliency = np.max(np.abs(gradients), axis=-1)[0]
        
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
        
        heatmap = create_heatmap(saliency, config)
        
        original_img = Image.fromarray((img_array[0] * 255).astype(np.uint8))
        original_img = original_img.resize(config.target_size, Image.Resampling.LANCZOS)
        
        return blend_images(original_img, heatmap, config)
        
    except Exception as e:
        logger.error(f"Error generating saliency map: {str(e)}")
        raise
