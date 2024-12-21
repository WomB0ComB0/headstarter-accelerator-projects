#! /usr/bin/env python3
# app/config.py
# -*- coding: utf-8 -*-

"""Configuration module for the Image Generation API.

This module provides configuration management through Pydantic models and environment variables.
It implements a singleton pattern for application-wide configuration access.

The configuration is structured into several components:
- Environment settings (development, production, testing)
- Redis configuration 
- Modal configuration for GPU compute
- Database configuration
- General application settings

Environment variables are loaded from a .env file in the application root directory.

Example usage:
    from .config import config
    
    # Access Redis configuration
    redis_url = config.redis.url
    
    # Check environment
    if config.is_production():
        # Production-specific logic
        pass
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field
import os
from enum import Enum

basedir = os.path.abspath(os.path.dirname(__file__))


class EnvironmentType(str, Enum):
    """Enum defining possible environment types for the application."""

    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"


class RedisConfig(BaseModel):
    """Configuration model for Redis connection settings.

    Attributes:
        url (str): Redis server URL
        token (str): Authentication token for Redis
    """

    url: str
    token: str


class ModalConfig(BaseModel):
    """Configuration model for Modal compute settings.

    Attributes:
        webhook_url (str): Webhook URL for Modal notifications
        api_key (str): Modal API authentication key
        timeout (int): Request timeout in seconds
        gpu_type (str): GPU type to use (A100, H100, or T4)
    """

    webhook_url: str
    api_key: str
    timeout: int
    gpu_type: Literal["A100", "H100", "T4"]


class DatabaseConfig(BaseModel):
    """Configuration model for database connection settings.

    Attributes:
        url (str): Database connection URL
        pool_size (int): Size of the connection pool
        max_overflow (int): Maximum number of connections that can be created beyond pool_size
    """

    url: str
    pool_size: int
    max_overflow: int


class Settings(BaseModel):
    """Main settings model containing all configuration options.

    Attributes:
        ENV (EnvironmentType): Current environment type
        DEBUG (bool): Debug mode flag
        APP_NAME (str): Application name
        APP_VERSION (str): Application version
        REDIS (RedisConfig): Redis configuration
        MODAL (ModalConfig): Modal compute configuration
        DATABASE (DatabaseConfig): Database configuration
        SECRET_KEY (str): Application secret key
        API_KEY (str): API authentication key
        ALLOWED_HOSTS (list[str]): List of allowed hosts
        MODEL_ID (str): ID of the ML model to use
        MAX_TOKENS (int): Maximum tokens for text processing
    """

    ENV: EnvironmentType = Field(
        default=EnvironmentType.DEVELOPMENT, description="Current environment"
    )
    DEBUG: bool = Field(default=False)

    APP_NAME: str = Field(default="Image Generation API")
    APP_VERSION: str = Field(default="1.0.0")

    REDIS: RedisConfig = Field(
        default={
            "url": os.environ["UPSTASH_REDIS_REST_URL"],
            "token": os.environ["UPSTASH_REDIS_REST_TOKEN"],
        }
    )

    MODAL: ModalConfig = Field(
        default={
            "webhook_url": os.environ["MODAL_WEBHOOK_URL"],
            "api_key": os.environ["MODAL_API_KEY"],
            "timeout": 30,
            "gpu_type": "A100",
        }
    )

    DATABASE: DatabaseConfig = Field(
        default={
            "url": "postgresql://user:pass@localhost:5432/db",
            "pool_size": 5,
            "max_overflow": 10,
        }
    )

    SECRET_KEY: str = Field(default=os.environ["SECRET_KEY"])
    API_KEY: str = Field(default=os.environ["API_KEY"])
    ALLOWED_HOSTS: list[str] = Field(default=["localhost", "127.0.0.1"])

    MODEL_ID: str = Field(default="runwayml/stable-diffusion-v1-5")
    MAX_TOKENS: int = Field(default=1000)

    class Config:
        """Pydantic configuration class."""

        env_file = os.path.join(basedir, ".env")
        env_file_encoding = "utf-8"
        case_sensitive = True


class AppConfig:
    """Singleton configuration class for application-wide settings.

    This class implements the singleton pattern to ensure only one instance
    of the configuration is created and used throughout the application.

    Attributes:
        _instance: Singleton instance
        _settings: Settings instance
    """

    _instance = None
    _settings: Optional[Settings] = None

    def __new__(cls):
        """Create or return the singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._settings = Settings()
        return cls._instance

    @property
    def settings(self) -> Settings:
        """Get the settings instance.

        Returns:
            Settings: The application settings
        """
        if not self._settings:
            self._settings = Settings()
        return self._settings

    @property
    def redis(self) -> RedisConfig:
        """Get Redis configuration.

        Returns:
            RedisConfig: Redis configuration settings
        """
        return self.settings.REDIS

    @property
    def modal(self) -> ModalConfig:
        """Get Modal configuration.

        Returns:
            ModalConfig: Modal configuration settings
        """
        return self.settings.MODAL

    @property
    def database(self) -> DatabaseConfig:
        """Get database configuration.

        Returns:
            DatabaseConfig: Database configuration settings
        """
        return self.settings.DATABASE

    def is_development(self) -> bool:
        """Check if running in development environment.

        Returns:
            bool: True if in development environment
        """
        return self.settings.ENV == EnvironmentType.DEVELOPMENT

    def is_production(self) -> bool:
        """Check if running in production environment.

        Returns:
            bool: True if in production environment
        """
        return self.settings.ENV == EnvironmentType.PRODUCTION


# Global configuration instance
config = AppConfig()
