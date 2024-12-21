#! /usr/bin/env python3
# app/rate_limit.py
# -*- coding: utf-8 -*-

"""
Rate limiting module for FastAPI applications.

This module provides rate limiting functionality using Redis as a backend store.
It includes:

- RateLimiter class for checking rate limits
- RateLimitMiddleware for global rate limiting
- rate_limit decorator for per-endpoint rate limiting

Rate limits are enforced using a sliding window algorithm with Redis sorted sets.
Each client is identified by either their IP address or auth token.

Example usage:

    # Global rate limiting
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, limit=100, window=60)

    # Per-endpoint rate limiting
    @app.get("/endpoint")
    @rate_limit(limit=10, window=60)
    async def endpoint():
        return {"message": "Rate limited endpoint"}
"""

from fastapi import Request, Response, HTTPException, Depends
from typing import Callable, Optional, Tuple, Dict
from time import time
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware import Middleware
from upstash_redis import Redis
from .config import config


class RateLimiter:
    """
    Rate limiter implementation using Redis as a backend.

    Attributes:
        redis (Redis): Redis client instance
        default_limit (int): Default requests per window (100)
        default_window (int): Default window size in seconds (60)
    """

    def __init__(self, redis_client: Optional[Redis] = None) -> None:
        """
        Initialize rate limiter with Redis client.

        Args:
            redis_client: Optional Redis client instance. If not provided,
                         creates new client using config settings.
        """
        self.redis = redis_client or Redis(
            url=config.redis["url"],
            token=config.redis["token"],
        )
        self.default_limit = 100
        self.default_window = 60  # seconds

    async def is_rate_limited(
        self, client_id: str, limit: int = None, window: int = None
    ) -> Tuple[bool, Dict]:
        """
        Check if the client is rate limited.

        Uses Redis sorted sets to implement a sliding window rate limit.
        Removes expired entries and counts requests within the window.

        Args:
            client_id: Unique identifier for the client
            limit: Maximum requests allowed in window (default: self.default_limit)
            window: Window size in seconds (default: self.default_window)

        Returns:
            Tuple containing:
                - bool: True if rate limited, False otherwise
                - dict: Rate limit information including limit, remaining requests, and reset time
        """
        limit = limit or self.default_limit
        window = window or self.default_window

        # Create Redis key for this client
        key = f"rate_limit:{client_id}"
        now = int(time())

        pipe = self.redis.pipeline()

        # Remove old entries
        pipe.zremrangebyscore(key, 0, now - window)
        # Add current request
        pipe.zadd(key, {str(now): now})
        # Count requests in window
        pipe.zcard(key)
        # Set expiry on bucket
        pipe.expire(key, window)

        _, _, request_count, _ = pipe.execute()

        remaining = max(0, limit - request_count)
        reset_time = now + window

        rate_limit_info = {"limit": limit, "remaining": remaining, "reset": reset_time}

        return request_count > limit, rate_limit_info


class RateLimitMiddleware(Middleware):
    """
    FastAPI middleware for global rate limiting.

    Applies rate limiting to all routes in the application.
    Adds rate limit headers to responses.
    """

    def __init__(
        self,
        app: FastAPI,
        limit: int = 100,
        window: int = 60,
        redis_client: Optional[Redis] = None,
    ) -> None:
        """
        Initialize middleware.

        Args:
            app: FastAPI application instance
            limit: Maximum requests per window
            window: Window size in seconds
            redis_client: Optional Redis client instance
        """
        super().__init__(app)
        self.limiter = RateLimiter(redis_client)
        self.limit = limit
        self.window = window

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request through rate limiter.

        Args:
            request: FastAPI request object
            call_next: Next middleware/handler in chain

        Returns:
            Response object with rate limit headers

        Raises:
            429 status code if rate limit exceeded
        """
        # Get client identifier (IP or auth token)
        client_id = request.client.host

        # Check for auth token
        auth_token = request.headers.get("Authorization")
        if auth_token:
            client_id = auth_token

        is_limited, rate_limit_info = await self.limiter.is_rate_limited(
            client_id, self.limit, self.window
        )

        # Add rate limit headers
        headers = {
            "X-RateLimit-Limit": str(rate_limit_info["limit"]),
            "X-RateLimit-Remaining": str(rate_limit_info["remaining"]),
            "X-RateLimit-Reset": str(rate_limit_info["reset"]),
        }

        if is_limited:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "retry_after": rate_limit_info["reset"] - int(time()),
                },
                headers=headers,
            )

        response = await call_next(request)

        # Add rate limit headers to response
        for key, value in headers.items():
            response.headers[key] = value

        return response


def rate_limit(limit: int = None, window: int = None) -> Callable:
    """
    Dependency for rate limiting specific endpoints.

    Can be used as a decorator to apply rate limiting to individual routes.

    Args:
        limit: Maximum requests per window
        window: Window size in seconds

    Returns:
        FastAPI dependency callable

    Example:
        @app.get("/endpoint")
        @rate_limit(limit=10, window=60)
        async def endpoint():
            return {"message": "Rate limited endpoint"}
    """

    async def rate_limit_dep(
        request: Request, limiter: RateLimiter = Depends(RateLimiter)
    ):
        client_id = request.client.host
        auth_token = request.headers.get("Authorization")
        if auth_token:
            client_id = auth_token

        is_limited, rate_limit_info = await limiter.is_rate_limited(
            client_id, limit, window
        )

        if is_limited:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Rate limit exceeded",
                    "retry_after": rate_limit_info["reset"] - int(time()),
                },
            )

        return rate_limit_info

    return rate_limit_dep
