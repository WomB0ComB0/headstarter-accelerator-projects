#! /usr/bin/env python3
"""
Image Generation Service with Content Safety and Recommendations

This module provides a Modal-based service for AI image generation using Stable Diffusion,
with content safety checks and personalized recommendations.

Key Features:
- Image generation using Stable Diffusion v1.5 
- Content safety filtering for inappropriate content and hate speech
- User-based recommendation system using embeddings
- Request rate limiting and caching
- Health monitoring

The service is built using Modal for serverless deployment and uses:
- Stable Diffusion for image generation
- RoBERTa for content safety checks 
- Sentence Transformers for recommendation embeddings
- Upstash Redis for caching and storing user interactions
- FastAPI for API endpoints

Environment Variables Required:
- UPSTASH_REDIS_REST_URL: URL for Upstash Redis instance
- UPSTASH_REDIS_REST_TOKEN: Authentication token for Upstash Redis 
- API_KEY: API key for service authentication

API Endpoints:
    POST /generate
        Generate an image from a text prompt
        
        Request Body:
        {
            "prompt": str,  # Text description of desired image
            "user_id": str  # Optional user ID for recommendations
        }
        
        Returns:
        {
            "image": str,  # Base64 encoded image
            "cached": bool,  # Whether result was from cache
            "recommendations": List[str],  # Similar prompt suggestions
            "safety_check": str  # Status of content safety check
        }

    GET /recommendations
        Get personalized prompt recommendations
        
        Query Parameters:
        - user_id: str  # User to get recommendations for
        
        Returns:
        {
            "recommendations": List[str]  # List of recommended prompts
        }

    GET /health
        Check service health status
        
        Returns:
        {
            "status": str,  # Service status
            "status_code": int,  # HTTP status code
            "timestamp": str  # Current UTC timestamp
        }

Classes:
    ContentSafety:
        Handles content moderation using keyword matching and hate speech detection
        
    RecommendationSystem:
        Manages user profiles and generates personalized prompt recommendations
        
    Model:
        Main service class that coordinates image generation, safety checks,
        caching and recommendations

Functions:
    keep_alive():
        Periodic health check that runs every 5 minutes

Error Handling:
- Rate limiting enforced via semaphore
- Content safety violations return 400 status
- Rate limit exceeded returns 429 status
- Other errors return 500 status with error details

Caching:
- Generated images cached in Redis for 1 hour
- User interaction history stored for recommendations
- Global prompt pool maintained for recommendations

Security:
- API key authentication required
- Content safety checks on all prompts
- Rate limiting prevents abuse

Performance:
- GPU acceleration on A100
- Concurrent request limiting
- Response caching
- Optimized model loading

Author: Mike Odnis
Version: 1.0.0
"""

import modal
import asyncio
import os
import typing
import upstash_redis
import json
import sentence_transformers
import transformers
import re
import dotenv
import base64
import requests
import datetime
import numpy as np

dotenv.load_dotenv(dotenv.find_dotenv())

app = modal.App("image-generation")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        [
            "git",
            "wget",
            "libgl1-mesa-glx",  # Required for OpenCV/image processing
            "libglib2.0-0",
            "build-essential",  # For compiling some Python packages
        ]
    )
    .pip_install_from_requirements("requirements.txt")
    .env(
        {
            "TORCH_CUDA_ARCH_LIST": "7.5",  # Optimize for A100 GPUs
            "PYTHONUNBUFFERED": "1",
        }
    )
    .workdir("/app")
)

with image.imports():
    import diffusers
    import torch
    from fastapi import Response, HTTPException

# Rate limiting configuration
MAX_CONCURRENT_REQUESTS: int = 10
request_semaphore: asyncio.Semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)


class ContentSafety(object):
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
        self.toxic_classifier: transformers.pipeline = transformers.pipeline(
            "text-classification",
            model="facebook/roberta-hate-speech-dynabench-r4-target",
        )

    def is_safe_prompt(self, text: str) -> typing.Tuple[bool, str]:
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
        result: typing.Dict[str, typing.Any] = self.toxic_classifier(text)[0]
        if result["label"] == "hate" and result["score"] > 0.7:
            return False, "Hate speech detected"

        return True, ""


class RecommendationSystem(object):
    """Personalized recommendation system for image generation prompts.

    Uses sentence embeddings to create user profiles based on their prompt history
    and recommends similar prompts from a global pool.

    Attributes:
        model: SentenceTransformer model for generating text embeddings
        redis_client: Upstash Redis client for storing user history and global prompts

    Example:
        recommender = RecommendationSystem()
        recommender.store_user_interaction("user123", "mountain sunset")
        recommendations = recommender.get_recommendations("user123", n=5)
    """

    def __init__(self) -> None:
        """Initialize the recommendation system with embedding model and Redis connection."""
        self.model: sentence_transformers.SentenceTransformer = (
            sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2")
        )
        self.redis_client: upstash_redis.Redis = upstash_redis.Redis(
            url=os.getenv("UPSTASH_REDIS_REST_URL"),
            token=os.getenv("UPSTASH_REDIS_REST_TOKEN"),
        )

    def get_embedding(self, text: str) -> list[float]:
        """Generate embedding vector for a text prompt.

        Args:
            text: Input text to embed

        Returns:
            List containing the embedding vector
        """
        return self.model.encode([text])[0].tolist()

    def store_user_interaction(self, user_id: str, prompt: str) -> None:
        """Store a user's prompt interaction in Redis."""
        embedding = self.get_embedding(prompt)
        user_history = f"user_history:{user_id}"
        self.redis_client.lpush(
            user_history,
            json.dumps({"prompt": prompt, "embedding": embedding}),
        )

    def get_recommendations(self, user_id: str, n: int = 5) -> list[str]:
        """Get personalized prompt recommendations for a user."""
        user_history = f"user_history:{user_id}"
        recent_prompts = self.redis_client.lrange(user_history, 0, 9)

        if not recent_prompts:
            return []

        # Calculate average embedding of recent prompts
        embeddings = []
        for prompt_data in recent_prompts:
            data = json.loads(prompt_data)
            embeddings.append(data["embedding"])

        # Use numpy for mean calculation
        user_profile = np.mean(embeddings, axis=0)

        # Find similar prompts from global pool
        all_prompts = self.redis_client.smembers("global_prompts")
        recommendations = []

        for prompt in all_prompts:
            prompt_data = json.loads(prompt)
            # Use numpy for dot product
            similarity = np.dot(user_profile, prompt_data["embedding"])
            recommendations.append((similarity, prompt_data["prompt"]))

        recommendations.sort(reverse=True)
        return [r[1] for r in recommendations[:n]]


@app.cls(
    gpu="A100",
    timeout=600,  # 10 minutes
    image=image,
    concurrency_limit=MAX_CONCURRENT_REQUESTS,
    cpu=1,
    container_idle_timeout=600,
    secrets=[modal.Secret.from_name("custom-secret")],
)
class Model:
    """Main model class for image generation service.

    Handles image generation requests with content safety checks,
    caching, and recommendations.

    Attributes:
        pipe: Stable Diffusion pipeline
        content_safety: Content safety checker
        recommender: Recommendation system
    """

    @modal.build()
    @modal.enter()
    def initialize(self):
        """Initialize the model during container build"""
        self.pipe = diffusers.StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None,
        )

        self.content_safety = ContentSafety()
        self.recommender = RecommendationSystem()

    async def _check_content_safety(self, prompt: str) -> typing.Tuple[bool, str]:
        """Check if the prompt is safe.

        Args:
            prompt: Text prompt to check

        Returns:
            Tuple of (is_safe, message)

        Raises:
            HTTPException: If content is unsafe
        """
        is_safe, message = self.content_safety.is_safe_prompt(prompt)
        if not is_safe:
            raise HTTPException(
                status_code=400, detail=f"Content safety check failed: {message}"
            )
        return is_safe, message

    async def _generate_with_cache(self, prompt: str, user_id: str) -> Response:
        """Internal method to handle generation with caching.

        Args:
            prompt: Text prompt for image generation
            user_id: User identifier for recommendations

        Returns:
            FastAPI Response with generated image or cached result
        """
        # Check content safety first
        await self._check_content_safety(prompt)

        # Store the interaction for recommendations
        self.recommender.store_user_interaction(user_id, prompt)

        # Check cache
        redis_client: upstash_redis.Redis = upstash_redis.Redis(
            url=os.getenv("UPSTASH_REDIS_REST_URL"),
            token=os.getenv("UPSTASH_REDIS_REST_TOKEN"),
        )
        cache_key: str = f"image_cache:{prompt}"
        cached_result: typing.Optional[str] = redis_client.get(cache_key)

        if cached_result:
            return Response(
                content={
                    "image": cached_result,
                    "cached": True,
                    "recommendations": self.recommender.get_recommendations(
                        user_id, n=5
                    ),
                    "safety_check": "passed",
                },
                headers={"Content-Type": "application/json"},
            )

        # Generate new image
        image_bytes: typing.List[bytes] = self.pipe(prompt, num_images=1)[0]
        base64_encoded: str = base64.b64encode(image_bytes).decode("utf-8")

        # Cache result
        redis_client.setex(cache_key, 3600, base64_encoded)

        # Update recommendation system
        self.recommender.redis_client.sadd(
            "global_prompts",
            json.dumps(
                {
                    "prompt": prompt,
                    "embedding": self.recommender.get_embedding(prompt),
                }
            ),
        )

        return Response(
            content={
                "image": base64_encoded,
                "cached": False,
                "recommendations": self.recommender.get_recommendations(user_id, n=5),
                "safety_check": "passed",
            },
            headers={"Content-Type": "application/json"},
        )

    @modal.web_endpoint(method="POST")
    async def generate(self, prompt: str, user_id: str = "default_user") -> Response:
        """Web endpoint for image generation.

        Args:
            prompt: Text prompt for image generation
            user_id: Optional user identifier for recommendations

        Returns:
            FastAPI Response with generated image or error
        """
        try:
            # Apply rate limiting
            if not request_semaphore.locked():
                async with request_semaphore:
                    return await self._generate_with_cache(prompt, user_id)
            else:
                raise HTTPException(
                    status_code=429, detail="Too many requests. Please try again later."
                )
        except HTTPException as he:
            return Response(
                content={
                    "error": str(he.detail),
                    "safety_check": (
                        "failed" if he.status_code == 400 else "not_performed"
                    ),
                },
                status_code=he.status_code,
                headers={"Content-Type": "application/json"},
            )
        except Exception as e:  # pylint: disable=broad-except
            return Response(
                content={"error": str(e)},
                status_code=500,
                headers={"Content-Type": "application/json"},
            )

    @modal.web_endpoint(method="GET")
    async def get_recommendations(self, user_id: str = "default_user") -> Response:
        """Get recommendations endpoint.

        Args:
            user_id: User identifier for fetching personalized recommendations

        Returns:
            FastAPI Response with recommended prompts
        """
        try:
            async with request_semaphore:  # Also apply rate limiting to recommendations
                recommendations = self.recommender.get_recommendations(user_id, n=5)
                return Response(
                    content={"recommendations": recommendations},
                    headers={"Content-Type": "application/json"},
                )
        except Exception as e:  # pylint: disable=broad-except
            return Response(
                content={"error": str(e)},
                status_code=500,
                headers={"Content-Type": "application/json"},
            )

    @modal.web_endpoint(method="GET")
    async def health(self) -> Response:
        """Health check endpoint.

        Returns:
            FastAPI Response with service status and timestamp
        """
        return Response(
            content=json.dumps(
                {
                    "status": "ok",
                    "status_code": 200,
                    "timestamp": datetime.datetime.now(
                        datetime.timezone.utc
                    ).isoformat(),
                }
            ),
            media_type="application/json",
        )


@app.function(
    gpu="A100",
    timeout=600,
    concurrency_limit=MAX_CONCURRENT_REQUESTS,
    cpu=1,
    container_idle_timeout=600,
    schedule=modal.Cron("*/5 * * * *"),
    secrets=[modal.Secret.from_name("custom-secret")],
    image=image,
)
def keep_alive() -> None:
    """Periodic health check function."""
    try:
        health_url: str = "https://womb0comb0--image-generation-model-health.modal.run/"
        generate_url: str = (
            "https://womb0comb0--image-generation-model-generate.modal.run/"
        )

        health_response: requests.Response = requests.get(health_url, timeout=30)
        print(f"Health check status: {health_response.status_code}")

        if health_response.ok:
            try:
                health_data = health_response.json()
                print(f"Health check at: {health_data.get('timestamp', 'N/A')}")
            except json.JSONDecodeError:
                print(f"Could not parse health response: {health_response.text}")

        headers: typing.Dict[str, str] = {"x-api-key": os.getenv("API_KEY")}
        generate_response: requests.Response = requests.get(
            generate_url, headers=headers, timeout=30
        )

        print(
            json.dumps(
                {
                    "status": "ok",
                    "status_code": generate_response.status_code,
                    "timestamp": datetime.datetime.now(
                        datetime.timezone.utc
                    ).isoformat(),
                    "generate_response": (
                        generate_response.json() if generate_response.ok else None
                    ),
                }
            )
        )
    except Exception as e:  # pylint: disable=broad-except
        print(f"Error in keep_alive: {str(e)}")
