#! /usr/bin/env python3
# app/routers/recommendations.py
# -*- coding: utf-8 -*-
"""Image Generation Router

This module implements the image generation endpoints using FastAPI. It provides
functionality for generating images from text prompts using Stable Diffusion models,
with features like content safety checks, rate limiting, and user recommendations.

Key Features:
    - GPU-accelerated image generation using Modal and Stable Diffusion
    - Content safety checks to prevent inappropriate content
    - Rate limiting to prevent abuse
    - Redis-based caching for performance
    - User-based recommendations for personalized prompts
    - Concurrent request handling with semaphores

Endpoints:
    POST /generate-image
        Generate an image from a text prompt
        Request Body: {"text": "prompt text"}
        Headers:
            - user-id (optional): For personalized recommendations
            - Authorization (optional): For rate limiting
        Returns: ImageResponse with generated image or error

Components:
    - Modal stub for GPU compute
    - Content safety checker
    - Recommendation system
    - Rate limiter
    - Redis caching

Configuration:
    - MAX_CONCURRENT_REQUESTS: Maximum concurrent requests (10)
    - GPU: A100 for image generation
    - Timeout: 2 seconds for low latency
    - Image size: 512x512 pixels
    - Cache duration: 1 hour

Dependencies:
    - FastAPI for API routing
    - Modal for GPU compute
    - Redis for caching
    - Stable Diffusion for image generation
    - Content safety and recommendation systems

Example Usage:
    POST /api/v1/generate-image
    {
        "text": "a beautiful sunset over mountains"
    }

Environment Variables Required:
    - REDIS_URL: URL for Redis connection
    - REDIS_TOKEN: Authentication token for Redis
"""

import os
import io
from fastapi import APIRouter, BackgroundTasks, Header
from pydantic import BaseModel
import modal
import asyncio
from ..utils.rate_limit import rate_limit
from ..utils.recommendations import RecommendationSystem
from ..utils.safety import ContentSafety

router: APIRouter = APIRouter()

# Initialize Modal stub for serverless GPU compute
stub: modal.Stub = modal.Stub("image-generation")
MAX_CONCURRENT_REQUESTS: int = 10
request_semaphore: asyncio.Semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

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
        from upstash_redis import Redis
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


@router.post("/generate-image", response_model=ImageResponse)
@rate_limit(limit=10, window=60)
async def create_image(
    request: ImageRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Header(None),
) -> ImageResponse:
    """Generate an image from a text prompt with safety checks and rate limiting.

    This endpoint handles image generation requests with several key features:
    - Content safety checking
    - Rate limiting (10 requests per minute)
    - Concurrent request management
    - User interaction tracking for recommendations
    - Error handling and response formatting

    Args:
        request (ImageRequest): Contains the text prompt for image generation
        background_tasks (BackgroundTasks): FastAPI background tasks for async operations
        user_id (str, optional): User identifier for tracking and recommendations

    Returns:
        ImageResponse: Contains either:
            - success=True with generated image bytes
            - success=False with error message

    Raises:
        HTTPException: For rate limiting violations
        Exception: Handled internally, returns error response

    Rate Limiting:
        - 10 requests per 60-second window
        - Enforced per user/IP
        - Returns 429 status when exceeded

    Concurrency:
        - Limited to 10 concurrent requests
        - Uses asyncio semaphore for management
        - Queues excess requests

    Example Success Response:
        {
            "success": true,
            "image": <bytes>,
            "error": null
        }

    Example Error Response:
        {
            "success": false,
            "image": null,
            "error": "Content safety check failed: explicit content detected"
        }
    """
    async with request_semaphore:
        try:
            # Check content safety
            (
                is_safe,
                message,
            ) = content_safety.is_safe_prompt(request.text)
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
