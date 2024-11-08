#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from functools import lru_cache
import os
from typing import Tuple, Any
import numpy as np
import cv2
import PIL.Image
import tensorflow as tf
import google.generative as genai
from dataclasses import dataclass

@dataclass
class ImageProcessingConfig:
    """Configuration for image processing parameters"""
    blur_kernel: Tuple[int, int] = (11, 11)
    threshold_percentile: float = 80
    heatmap_alpha: float = 0.7
    original_alpha: float = 0.3

@lru_cache(maxsize=1)
def load_xception_model(path: str) -> tf.keras.Model:
    """Load and cache the Xception model"""
    img_shape = (299, 299, 3)
    base_model = tf.keras.applications.Xception(
        input_shape=img_shape,
        include_top=False,
        weights="imagenet",
        pooling="max",
    )
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(4, activation="softmax"),
    ])
    
    model.build((None,) + img_shape)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )
    
    model.load_weights(path)
    return model

def create_circular_mask(img_size: Tuple[int, int]) -> np.ndarray:
    """Create a circular mask for the brain region"""
    center = (img_size[0] // 2, img_size[1] // 2)
    radius = min(center[0], center[1]) - 10
    y, x = np.ogrid[:img_size[0], :img_size[1]]
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
    img_size: Tuple[int, int],
    img: PIL.Image.Image,
    upload_file: Any,
    output_dir: str,
    config: ImageProcessingConfig = ImageProcessingConfig()
) -> np.ndarray:
    """Generate saliency map with improved gradient processing"""
    
    # Calculate gradients
    with tf.GradientTape() as tape:
        img_tensor = tf.convert_to_tensor(img_array)
        tape.watch(img_tensor)
        predictions = model(img_tensor)
        target_class = predictions[:, class_index]
    
    # Process gradients
    gradients = tf.abs(tape.gradient(target_class, img_tensor))
    gradients = tf.reduce_max(gradients, axis=-1).numpy().squeeze()
    gradients = cv2.resize(gradients, img_size)
    
    # Apply circular mask
    mask = create_circular_mask(img_size)
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
    heatmap = cv2.resize(heatmap, img_size)
    
    # Combine with original image
    original_img = np.array(img)
    superimposed_img = (heatmap * config.heatmap_alpha + 
                       original_img * config.original_alpha)
    superimposed_img = superimposed_img.astype(np.uint8)
    
    # Save results
    img_path = os.path.join(output_dir, upload_file.name)
    with open(img_path, "wb") as f:
        f.write(upload_file.getbuffer())
    
    saliency_map_path = os.path.join(output_dir, f"saliency_maps/{upload_file.name}")
    cv2.imwrite(saliency_map_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))
    
    return superimposed_img

def generate_explanation(
    img_path: str,
    model_prediction: str,
    confidence: float,
) -> str:
    """Generate explanation using Gemini with improved prompt"""
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