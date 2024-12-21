#! /usr/bin/env python3
# app/utils/recommendations.py
# -*- coding: utf-8 -*-

from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from upstash_redis import Redis
import json
import os


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
