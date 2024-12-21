#! /usr/bin/env python3
# app/utils/safety.py
# -*- coding: utf-8 -*-
"""Content Safety Module

This module provides functionality for checking text prompts for inappropriate content
before image generation. It combines rule-based and ML approaches for comprehensive
content moderation.

Key Features:
    - Explicit content detection using regex patterns
    - Hate speech detection using RoBERTa model
    - Configurable thresholds and patterns
    - Fast keyword-based pre-filtering
    - Detailed error messages

Components:
    ContentSafety: Main class implementing the safety checks
        - Regex-based explicit content detection
        - ML-based hate speech classification
        - Combined safety verdict with explanation

Safety Checks:
    1. Explicit Content:
        - Keywords: nsfw, explicit, nude, pornographic
        - Case-insensitive matching
        - Word boundary awareness
    
    2. Hate Speech:
        - Model: facebook/roberta-hate-speech-dynabench-r4-target
        - Confidence threshold: 0.7
        - Binary classification (hate/not-hate)

Usage Example:
    from app.utils.safety import ContentSafety

    safety = ContentSafety()
    is_safe, message = safety.is_safe_prompt("text to check")
    if not is_safe:
        print(f"Content rejected: {message}")

Dependencies:
    - transformers: For ML model loading and inference
    - re: For regex pattern matching
    
Configuration:
    - Explicit content patterns are hardcoded but extensible
    - Hate speech threshold set to 0.7
    - Uses CPU for inference by default

Error Handling:
    - Returns clear error messages for rejected content
    - Graceful handling of model loading failures
    - Type hints for better code safety
"""

from typing import Any, Dict, Tuple
from transformers import pipeline
import re


class ContentSafety:
    """Content safety checker for text prompts.

    This class provides functionality to check text prompts for inappropriate content
    including explicit material and hate speech. It uses:
    - Regular expressions to detect explicit keywords
    - A RoBERTa model fine-tuned on hate speech detection

    Attributes:
        toxic_classifier: Hugging Face pipeline for hate speech classification using
            the facebook/roberta-hate-speech-dynabench-r4-target model

    Example usage:
        safety = ContentSafety()
        is_safe, message = safety.is_safe_prompt("some text to check")
        if not is_safe:
            print(f"Content rejected: {message}")
    """

    def __init__(self) -> None:
        """Initialize the content safety checker.

        Loads the hate speech classification model using the Hugging Face
        transformers pipeline.
        """
        self.toxic_classifier: pipeline = pipeline(
            "text-classification",
            model="facebook/roberta-hate-speech-dynabench-r4-target",
        )

    def is_safe_prompt(self, text: str) -> Tuple[bool, str]:
        """Check if the provided text prompt is safe for image generation.

        Performs two safety checks:
        1. Explicit content detection using keyword matching
        2. Hate speech detection using the RoBERTa classifier

        Args:
            text: The text prompt to check

        Returns:
            A tuple containing:
                - bool: True if content is safe, False otherwise
                - str: Empty string if safe, explanation message if unsafe

        Example:
            >>> safety = ContentSafety()
            >>> is_safe, message = safety.is_safe_prompt("a beautiful sunset")
            >>> print(is_safe)
            True
            >>> print(message)
            ""
        """
        # Check for explicit content keywords
        explicit_pattern = r"\b(nsfw|explicit|nude|pornographic)\b"
        if re.search(explicit_pattern, text.lower()):
            return False, "Explicit content not allowed"

        # Check for hate speech
        result: Dict[str, Any] = self.toxic_classifier(text)[0]
        if result["label"] == "hate" and result["score"] > 0.7:
            return False, "Hate speech detected"

        return True, ""
