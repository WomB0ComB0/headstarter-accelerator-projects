#! /usr/bin/env python3
# app/main.py
# -*- coding: utf-8 -*-
"""Image Generation API

This module implements a FastAPI application that provides image generation capabilities
using Stable Diffusion models. It includes comprehensive features for production use.

Key Features:
    - GPU-accelerated image generation using Modal and Stable Diffusion
    - Rate limiting and concurrency control for API stability
    - Content safety checks to prevent inappropriate content
    - Redis-based caching for performance optimization
    - User-based recommendations for personalized prompts
    - CORS middleware for frontend integration
    - Health check endpoint for monitoring
    - Structured logging

API Endpoints:
    POST /api/v1/generate-image
        Generate an image from a text prompt
        Request Body: {"text": "prompt text"}
        Headers: 
            - user-id (optional): For personalized recommendations
            - Authorization (optional): For rate limiting

    GET /api/v1/recommendations
        Get personalized prompt recommendations
        Headers:
            - user-id: Required for fetching recommendations

    GET /health
        Health check endpoint
        Returns: {"status": "ok"}

    GET /
        Root endpoint
        Returns: {"message": "Hello World"}

Configuration:
    The application uses environment variables for configuration:
    - REDIS_URL: URL for Redis connection
    - REDIS_TOKEN: Authentication token for Redis
    - PORT: Server port (default: 8000)
    - Additional Modal and safety-related configurations

Rate Limiting:
    - Global rate limiting across all endpoints
    - Per-endpoint customizable limits
    - Sliding window algorithm using Redis
    - Headers: X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset

Concurrency:
    - Semaphore-based request limiting
    - Configurable maximum concurrent requests
    - Graceful handling of overload scenarios

Error Handling:
    - Structured error responses
    - Comprehensive logging
    - Safe error propagation

Dependencies:
    - FastAPI: Web framework
    - Modal: Serverless GPU compute
    - Redis: Caching and rate limiting
    - uvicorn: ASGI server
    - logging: Structured logging
    - Other utility modules for safety and recommendations

Example Usage:
    # Start the server
    $ python -m app.main

    # Generate an image
    curl -X POST http://localhost:8000/api/v1/generate-image \
         -H "Content-Type: application/json" \
         -H "user-id: user123" \
         -d '{"text": "a beautiful sunset over mountains"}'

    # Get recommendations
    curl http://localhost:8000/api/v1/recommendations \
         -H "user-id: user123"
"""
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from .routers.generate_image import router as create_image
from .routers.recommendations import router as get_recommendations
from .config.config import config
from typing import Dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)

PREFIX: str = "/api/v1"

# Initialize FastAPI app with CORS middleware
app: FastAPI = FastAPI(title="Image Generation API")

app.include_router(create_image, prefix=PREFIX)
app.include_router(get_recommendations, prefix=PREFIX)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/")
async def root() -> Dict[str, str]:
    return {"message": "Hello World"}


if __name__ == "__main__":
    port = config.port or 8000
    uvicorn.run(app, host="0.0.0.0", port=port)
