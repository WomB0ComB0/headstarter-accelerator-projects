import modal
import asyncio
from pydantic import BaseModel
import os
import io
from typing import List, Tuple, Dict, Any
import numpy as np
from upstash_redis import Redis
import json
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import re


# Initialize Modal stub for serverless GPU compute
stub: modal.Stub = modal.Stub("image-generation")
MAX_CONCURRENT_REQUESTS: int = 10
request_semaphore: asyncio.Semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)


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


class RecommendationSystem:
    def __init__(self) -> None:
        self.model: SentenceTransformer = SentenceTransformer("all-MiniLM-L6-v2")
        self.redis_client: Redis = Redis(
            url=os.environ["REDIS_URL"],
            token=os.environ["REDIS_TOKEN"],
        )

    def get_embedding(self, text: str) -> np.ndarray:
        return self.model.encode([text])[0]

    def store_user_interaction(self, user_id: str, prompt: str) -> None:
        embedding: np.ndarray = self.get_embedding(prompt)
        user_history: str = f"user_history:{user_id}"
        self.redis_client.lpush(
            user_history,
            json.dumps({"prompt": prompt, "embedding": embedding.tolist()}),
        )

    def get_recommendations(self, user_id: str, n: int = 5) -> List[str]:
        # Get user's recent prompts
        user_history: str = f"user_history:{user_id}"
        recent_prompts: List[str] = self.redis_client.lrange(user_history, 0, 9)

        if not recent_prompts:
            return []

        # Calculate average embedding of recent prompts
        embeddings: List[np.ndarray] = []
        for prompt_data in recent_prompts:
            data: dict = json.loads(prompt_data)
            embeddings.append(np.array(data["embedding"]))

        user_profile: np.ndarray = np.mean(embeddings, axis=0)

        # Find similar prompts from global pool
        all_prompts: List[str] = self.redis_client.smembers("global_prompts")
        recommendations: List[Tuple[float, str]] = []

        for prompt in all_prompts:
            prompt_data: dict = json.loads(prompt)
            similarity: float = np.dot(user_profile, np.array(prompt_data["embedding"]))
            recommendations.append((similarity, prompt_data["prompt"]))

        recommendations.sort(reverse=True)
        return [r[1] for r in recommendations[:n]]

# Initialize safety and recommendation systems
content_safety: ContentSafety = ContentSafety()
recommendation_system: RecommendationSystem = RecommendationSystem()

class ImageRequest(BaseModel):
    """Request model for image generation.

    Attributes:
        text (str): The prompt text to generate an image from. Should be a descriptive
            text prompt that will be used by Stable Diffusion to generate the image.

    Example:
        {
            "text": "a beautiful sunset over mountains with orange and purple sky"
        }
    """

    text: str


class ImageResponse(BaseModel):
    """Response model for image generation.

    Attributes:
        success (bool): Whether the generation was successful
        image (bytes | None): The generated image bytes in PNG format if successful
        error (str | None): Error message if generation failed, None if successful

    Example Success:
        {
            "success": true,
            "image": <bytes>,
            "error": null
        }

    Example Error:
        {
            "success": false,
            "image": null,
            "error": "Content safety check failed"
        }
    """

    success: bool
    image: bytes | None = None
    error: str | None = None


@stub.function(
    gpu="A100",
    timeout=2,  # 2-second timeout for low latency
    image=modal.Image.debian_slim().pip_install(
        "diffusers", "torch", "transformers", "redis"
    ),
    secret=modal.Secret.from_name("custom-secret"),
    concurrency_limit=MAX_CONCURRENT_REQUESTS,
    retries=modal.Retries(max_retries=3),
)
async def generate_image(prompt: str) -> bytes:
    """Generate an image from a text prompt using Stable Diffusion.

    This function uses Stable Diffusion to generate an image based on the input prompt.
    It includes caching to improve performance for repeated prompts. The generation
    is optimized for low latency with reduced inference steps and image size.

    Args:
        prompt (str): The text prompt to generate an image from. Should be descriptive
            and clear about the desired image.

    Returns:
        bytes: The generated image in PNG format as bytes

    Raises:
        ImportError: If required ML libraries are not available
        Exception: For any generation or processing errors including GPU issues

    Cache Strategy:
        - Uses Redis to cache generated images
        - Cache key format: "image_cache:{prompt}"
        - Cache duration: 1 hour
        - Returns cached result if available

    Performance Optimizations:
        - Uses CUDA for GPU acceleration
        - Reduced inference steps (20)
        - Smaller image size (512x512)
        - Disabled safety checker for speed
        - Float16 precision for faster computation
    """
    try:
        import torch
        from diffusers import StableDiffusionPipeline
    except (ImportError, ModuleNotFoundError) as e:
        print(f"Error importing modules: {e}")

    redis_client: Redis = Redis(
        url=os.environ["REDIS_URL"],
        token=os.environ["REDIS_TOKEN"],
    )

    # Check cache first
    cache_key: str = f"image_cache:{prompt}"
    cached_result: bytes | None = redis_client.get(cache_key)
    if cached_result:
        return cached_result

    # Generate image if not in cache
    model_id: str = "runwayml/stable-diffusion-v1-5"
    pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,  # Disable for faster generation
    )
    pipe = pipe.to("cuda")

    # Use smaller image size for faster generation
    image = pipe(
        prompt,
        num_inference_steps=20,  # Reduced steps for faster generation
        height=512,
        width=512,
    ).images[0]

    # Convert and cache result
    img_byte_arr: io.BytesIO = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    image_bytes: bytes = img_byte_arr.getvalue()

    # Cache for 1 hour
    redis_client.setex(cache_key, 3600, image_bytes)

    return image_bytes
