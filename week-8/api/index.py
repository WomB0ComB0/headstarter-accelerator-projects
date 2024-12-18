"""
    @file: api/index.py
    @author: Mike Odnis
    @date: 2024-12-18
    @description: Main entry point for the API
"""

from requests import Response
from flask import Flask, request, jsonify
from flask_cors import CORS
from time import time
import os
import logging
from typing import Tuple, Dict, List
from config import Config
from functools import lru_cache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

CORS(
    app,
    resources={
        r"/api/*": {
            "origins": ["http://localhost:3000"],
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"],
        }
    },
)


class RateLimiter:
    def __init__(self, limit=100, window=60) -> None:
        self.limit = limit
        self.window = window
        self.tokens = {}

    def is_allowed(self, client_id):
        now = time()
        if client_id not in self.tokens:
            self.tokens[client_id] = []

        self.tokens[client_id] = [
            t for t in self.tokens[client_id] if t > now - self.window
        ]

        if len(self.tokens[client_id]) < self.limit:
            self.tokens[client_id].append(now)
            return True
        return False


limiter = RateLimiter()


def rate_limit_middleware() -> Tuple[Response, int]:
    client_id = request.remote_addr
    if not limiter.is_allowed(client_id):
        return (
            jsonify(
                {
                    "error": "Rate limit exceeded",
                    "message": "Too many requests, please try again later",
                }
            ),
            429,
        )


@app.route("/api/health")
def hello_world() -> Response:
    rate_limit_result = rate_limit_middleware()
    if rate_limit_result is not None:
        return rate_limit_result

    return (jsonify({"message": "OK!"}), 200)


@app.errorhandler(404)
def not_found(_e) -> Tuple[Response, int]:
    return (
        jsonify(
            {"error": "Not Found", "message": "The requested resource was not found"}
        ),
        404,
    )


@app.errorhandler(500)
def server_error(_e) -> Tuple[Response, int]:
    return (
        jsonify(
            {
                "error": "Internal Server Error",
                "message": "An unexpected error occurred",
            }
        ),
        500,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV") == "development"

    app.run(host="0.0.0.0", port=port, debug=debug)
