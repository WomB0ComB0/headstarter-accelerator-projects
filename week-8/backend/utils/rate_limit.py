#! /usr/bin/env python3
# app/utils/rate_limit.py
# -*- coding: utf-8 -*-

"""Rate limiting module for FastAPI applications.

This module provides rate limiting functionality using Redis as a backend store.
It implements a sliding window rate limiting algorithm to control request rates
on both a global and per-endpoint basis.

Key Features:
    - Redis-backed sliding window rate limiting
    - Global rate limiting via middleware
    - Per-endpoint rate limiting via decorator
    - Configurable limits and time windows
    - Rate limit headers in responses
    - Client identification via IP or auth token
    - Pipeline operations for atomic updates
    - Automatic cleanup of expired entries

Components:
    RateLimiter:
        Core rate limiting implementation using Redis sorted sets
        - Sliding window algorithm
        - Atomic operations with pipelining
        - Configurable limits and windows
        - Automatic cleanup of old entries

    RateLimitMiddleware:
        FastAPI middleware for global rate limiting
        - Applies to all routes
        - Adds rate limit headers
        - Returns 429 status on limit exceeded
        - Configurable limits per middleware instance

    rate_limit:
        Decorator for per-endpoint rate limiting
        - Individual route protection
        - Customizable limits per endpoint
        - Raises HTTPException on limit exceeded
        - Returns rate limit info to route

Usage Examples:
    Global Rate Limiting:
        ```python
        app = FastAPI()
        app.add_middleware(
            RateLimitMiddleware,
            limit=100,  # requests
            window=60   # seconds
        )
        ```

    Per-Endpoint Rate Limiting:
        ```python
        @app.get("/api/resource")
        @rate_limit(limit=10, window=60)
        async def get_resource():
            return {"data": "rate limited resource"}
        ```

    Custom Redis Client:
        ```python
        redis_client = Redis(url="redis://...", token="...")
        limiter = RateLimiter(redis_client=redis_client)
        ```

Rate Limit Headers:
    All responses include rate limit information:
    - X-RateLimit-Limit: Max requests per window
    - X-RateLimit-Remaining: Requests remaining
    - X-RateLimit-Reset: Window reset timestamp

Client Identification:
    Clients are identified by either:
    - IP address (request.client.host)
    - Authorization header token if present

Dependencies:
    - FastAPI for web framework integration
    - Redis for rate limit storage
    - upstash_redis for Redis client
    - time for timestamps

Configuration:
    Redis connection settings are loaded from config:
    - url: Redis server URL
    - token: Authentication token

Error Handling:
    Rate limit exceeded:
    - Status: 429 Too Many Requests
    - Response includes:
        - error: "Rate limit exceeded"
        - retry_after: Seconds until reset

Performance Considerations:
    - Uses Redis sorted sets for efficient sliding windows
    - Pipeline operations for atomic updates
    - Automatic cleanup of expired entries
    - Minimal memory footprint per client
"""

from fastapi import Request, Response, HTTPException, Depends
from typing import Callable, Optional, Tuple, Dict
from time import time
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware import Middleware
from upstash_redis import Redis
from backend.config.config import config


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
        self.redis: Redis = redis_client or Redis(
            url=config.redis["url"],
            token=config.redis["token"],
        )
        self.default_limit: int = 100
        self.default_window: int = 60  # seconds

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
        limit: int = limit or self.default_limit
        window: int = window or self.default_window

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
        self.limiter: RateLimiter = RateLimiter(redis_client)
        self.limit: int = limit
        self.window: int = window

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
        client_id: str = request.client.host

        # Check for auth token
        auth_token: str = request.headers.get("Authorization")
        if auth_token:
            client_id = auth_token

        (
            is_limited,
            rate_limit_info,
        ) = await self.limiter.is_rate_limited(client_id, self.limit, self.window)

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

        response: Response = await call_next(request)

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
        client_id: str = request.client.host
        auth_token: str = request.headers.get("Authorization")
        if auth_token:
            client_id = auth_token

        (
            is_limited,
            rate_limit_info,
        ) = await limiter.is_rate_limited(client_id, limit, window)

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
