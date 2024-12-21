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

    def __init__(self):
        """Initialize the content safety checker.

        Loads the hate speech classification model using the Hugging Face
        transformers pipeline.
        """
        self.toxic_classifier = pipeline(
            "text-classification",
            model="facebook/roberta-hate-speech-dynabench-r4-target",
        )

    def is_safe_prompt(self, text: str) -> tuple[bool, str]:
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
        result = self.toxic_classifier(text)[0]
        if result["label"] == "hate" and result["score"] > 0.7:
            return False, "Hate speech detected"

        return True, ""
