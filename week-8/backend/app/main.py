#! /usr/bin/env python3
# app/main.py
# -*- coding: utf-8 -*-
"""Image Generation API

This module implements a FastAPI application that provides image generation capabilities
using Stable Diffusion models. It includes:

- Rate limiting and concurrency control
- Content safety checks
- Redis-based caching
- User-based recommendations
- GPU-accelerated image generation using Modal

Key Components:
    - FastAPI application with CORS middleware
    - Modal stub for GPU-accelerated image generation
    - Content safety checking
    - Recommendation system for personalized prompts
    - Redis caching for generated images
    - Concurrent request handling with semaphores

Example Usage:
    POST /api/generate-image
    {
        "text": "a beautiful sunset over mountains"
    }

    GET /api/recommendations
    Header: user-id: <user_id>

Environment Variables Required:
    - REDIS_URL: URL for Redis connection
    - REDIS_TOKEN: Authentication token for Redis
    - Other Modal and safety-related configs
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import modal
import asyncio
import os
import io
from .safety import ContentSafety
from .recommendations import RecommendationSystem
from .rate_limit import rate_limit
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Initialize Modal stub for serverless GPU compute
stub = modal.Stub("image-generation")
MAX_CONCURRENT_REQUESTS = 10
request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# Initialize FastAPI app with CORS middleware
app = FastAPI(title="Image Generation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize safety and recommendation systems
content_safety = ContentSafety()
recommendation_system = RecommendationSystem()


class ImageRequest(BaseModel):
    """Request model for image generation.

    Attributes:
        text (str): The prompt text to generate an image from
    """

    text: str


class ImageResponse(BaseModel):
    """Response model for image generation.

    Attributes:
        success (bool): Whether the generation was successful
        image (bytes | None): The generated image bytes if successful
        error (str | None): Error message if generation failed
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
)
async def generate_image(prompt: str) -> bytes:
    """Generate an image from a text prompt using Stable Diffusion.

    Args:
        prompt (str): The text prompt to generate an image from

    Returns:
        bytes: The generated image in PNG format

    Raises:
        ImportError: If required dependencies are not available
        Exception: For any generation or processing errors
    """
    try:
        import torch
        from diffusers import StableDiffusionPipeline
        from upstash_redis import Redis
    except (ImportError, ModuleNotFoundError) as e:
        print(f"Error importing modules: {e}")

    redis_client = Redis(
        url=os.environ["REDIS_URL"],
        token=os.environ["REDIS_TOKEN"],
    )

    # Check cache first
    cache_key = f"image_cache:{prompt}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return cached_result

    # Generate image if not in cache
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
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
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    image_bytes = img_byte_arr.getvalue()

    # Cache for 1 hour
    redis_client.setex(cache_key, 3600, image_bytes)

    return image_bytes


@app.post("/api/generate-image", response_model=ImageResponse)
@rate_limit(limit=10, window=60)
async def create_image(
    request: ImageRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Header(None),
) -> ImageResponse:
    """Generate an image from a text prompt.

    Args:
        request (ImageRequest): The request containing the prompt text
        background_tasks (BackgroundTasks): FastAPI background tasks handler
        user_id (str, optional): User identifier for recommendations

    Returns:
        ImageResponse: Contains the generated image or error information

    Raises:
        HTTPException: For rate limiting or other API errors
    """
    async with request_semaphore:
        try:
            # Check content safety
            is_safe, message = content_safety.is_safe_prompt(request.text)
            if not is_safe:
                return ImageResponse(success=False, error=message)

            # Generate image
            image_bytes = await generate_image.remote(request.text)

            # Store interaction for recommendations
            if user_id:
                background_tasks.add_task(
                    recommendation_system.store_user_interaction, user_id, request.text
                )

            return ImageResponse(success=True, image=image_bytes)
        except Exception as e:  # pylint: disable=broad-except
            return ImageResponse(success=False, error=str(e))


@app.get("/api/recommendations")
@rate_limit(limit=10, window=60)
async def get_recommendations(user_id: str = Header(None)):
    """Get personalized prompt recommendations for a user.

    Args:
        user_id (str): User identifier to get recommendations for

    Returns:
        dict: Contains list of recommended prompts

    Raises:
        HTTPException: If user_id is not provided
    """
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID required")

    recommendations = recommendation_system.get_recommendations(user_id)
    return {"recommendations": recommendations}
