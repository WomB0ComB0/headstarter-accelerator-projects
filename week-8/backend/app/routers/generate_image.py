#! /usr/bin/env python3
# app/routers/generate_image.py
# -*- coding: utf-8 -*-
"""Image Generation Router

This module implements the recommendation endpoint for the image generation API.
It provides personalized prompt recommendations based on user history and preferences.

Key Features:
    - User-based recommendations using embeddings
    - Rate limiting to prevent abuse
    - Header-based user identification
    - FastAPI routing and error handling

Endpoints:
    GET /recommendations
        Get personalized prompt recommendations
        Headers:
            - user-id: Required for fetching recommendations
        Returns: List of recommended prompts
        Rate limit: 10 requests per 60 seconds

Components:
    - RecommendationSystem for generating personalized suggestions
    - Rate limiter for API stability
    - FastAPI router for endpoint handling

Configuration:
    - Rate limit: 10 requests per minute
    - User identification via headers
    - Error handling for missing user ID

Dependencies:
    - FastAPI for API routing
    - Custom rate limiting decorator
    - RecommendationSystem for generating suggestions

Example Usage:
    GET /api/v1/recommendations
    Headers:
        user-id: user123
    
    Response:
    {
        "recommendations": [
            "a beautiful sunset over mountains",
            "abstract art with vibrant colors",
            ...
        ]
    }
"""

from fastapi import APIRouter, HTTPException, Header
from ..utils.rate_limit import rate_limit
from ..utils.recommendations import RecommendationSystem

router = APIRouter()

recommendation_system = RecommendationSystem()


@router.get("/recommendations")
@rate_limit(limit=10, window=60)
async def get_recommendations(user_id: str = Header(None)):
    """Get personalized prompt recommendations for a user.

    This endpoint returns a list of recommended prompts based on the user's
    previous interactions and preferences. It uses embeddings to find similar
    prompts that the user might be interested in.

    Args:
        user_id (str): User identifier to get recommendations for. Must be
            provided as a header.

    Returns:
        dict: Contains list of recommended prompts in the format:
            {
                "recommendations": [
                    "prompt 1",
                    "prompt 2",
                    ...
                ]
            }

    Raises:
        HTTPException: 400 status code if user_id header is not provided

    Rate Limits:
        - 10 requests per 60 seconds per user
        - Rate limit headers included in response

    Example:
        >>> # With valid user ID
        >>> response = await get_recommendations(user_id="user123")
        >>> print(response)
        {"recommendations": ["sunset prompt", "nature prompt", ...]}

        >>> # Without user ID
        >>> try:
        >>>     await get_recommendations()
        >>> except HTTPException as e:
        >>>     print(e.detail)
        "User ID required"
    """
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID required")

    recommendations = recommendation_system.get_recommendations(user_id)
    return {"recommendations": recommendations}
