#! /usr/bin/env python3
# app/config.py
# -*- coding: utf-8 -*-

"""Configuration module for the Image Generation API.

This module provides configuration management through Pydantic models and environment variables.
It implements a singleton pattern for application-wide configuration access.

The configuration is structured into several components:
- Environment settings (development, production, testing)
- Redis configuration for caching and rate limiting
- Modal configuration for GPU compute and serverless functions
- Database configuration for data persistence
- General application settings like ports, keys, and model parameters

Key Features:
    - Environment-based configuration using Pydantic models
    - Type-safe configuration with validation
    - Singleton pattern for consistent access
    - Environment variable loading from .env files
    - Hierarchical configuration structure
    - Comprehensive documentation

Configuration Components:
    EnvironmentType:
        Enum defining possible runtime environments:
        - DEVELOPMENT: For local development
        - PRODUCTION: For production deployment
        - TESTING: For test environments

    RedisConfig:
        Redis connection settings:
        - url: Redis server URL
        - token: Authentication token

    ModalConfig:
        Modal compute settings:
        - webhook_url: Webhook URL for notifications
        - api_key: API authentication key
        - timeout: Request timeout in seconds
        - gpu_type: GPU type (A100, H100, T4)

    DatabaseConfig:
        Database connection settings:
        - url: Database connection URL
        - pool_size: Connection pool size
        - max_overflow: Max extra connections

    Settings:
        Main settings container:
        - Environment configuration
        - Debug settings
        - Application metadata
        - Component configurations
        - Security settings
        - Model parameters

Usage Examples:
    Basic Usage:
        from .config import config
        
        # Access Redis configuration
        redis_url = config.redis.url
        
        # Check environment
        if config.is_production():
            # Production-specific logic
            pass

    Environment Settings:
        # Get current environment
        env = config.settings.ENV
        
        # Check debug mode
        debug = config.settings.DEBUG

    Component Configuration:
        # Redis settings
        redis_config = config.redis
        
        # Modal settings
        modal_config = config.modal
        
        # Database settings
        db_config = config.database

    Application Settings:
        # Get port number
        port = config.port
        
        # Get model ID
        model_id = config.settings.MODEL_ID

Environment Variables:
    Required variables:
    - UPSTASH_REDIS_REST_URL: Redis connection URL
    - UPSTASH_REDIS_REST_TOKEN: Redis authentication token
    - MODAL_WEBHOOK_URL: Modal webhook URL
    - MODAL_API_KEY: Modal API key
    - SECRET_KEY: Application secret key
    - API_KEY: API authentication key

    Optional variables:
    - PORT: Server port (default: 8000)
    - DEBUG: Debug mode flag (default: False)
    - ENV: Environment type (default: development)

File Structure:
    .env - Environment variables file
    config.py - This configuration module
    
Dependencies:
    - pydantic: For data validation and settings management
    - python-dotenv: For .env file loading
    - typing: For type hints
    - os: For environment and path operations
    - enum: For environment type enumeration
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field
import os
from enum import Enum

basedir = os.path.abspath(os.path.dirname(__file__))


class EnvironmentType(str, Enum):
    """Enum defining possible environment types for the application.

    Attributes:
        DEVELOPMENT: Local development environment
        PRODUCTION: Production deployment environment
        TESTING: Testing environment
    """

    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"


class RedisConfig(BaseModel):
    """Configuration model for Redis connection settings.

    Attributes:
        url (str): Redis server URL for connection
        token (str): Authentication token for Redis access
    """

    url: str
    token: str


class ModalConfig(BaseModel):
    """Configuration model for Modal compute settings.

    Attributes:
        webhook_url (str): URL for Modal service webhooks
        api_key (str): Authentication key for Modal API
        timeout (int): Request timeout duration in seconds
        gpu_type (str): Type of GPU to use (A100, H100, or T4)
    """

    webhook_url: str
    api_key: str
    timeout: int
    gpu_type: Literal["A100", "H100", "T4"]


class DatabaseConfig(BaseModel):
    """Configuration model for database connection settings.

    Attributes:
        url (str): Database connection URL with credentials
        pool_size (int): Number of connections to maintain in pool
        max_overflow (int): Maximum additional connections allowed
    """

    url: str
    pool_size: int
    max_overflow: int


class Settings(BaseModel):
    """Main settings model containing all configuration options.

    This class serves as the central configuration container for all application
    settings. It includes environment settings, component configurations, and
    application parameters.

    Attributes:
        ENV (EnvironmentType): Current environment type
        DEBUG (bool): Debug mode flag
        APP_NAME (str): Name of the application
        APP_VERSION (str): Application version string
        REDIS (RedisConfig): Redis connection configuration
        MODAL (ModalConfig): Modal compute configuration
        DATABASE (DatabaseConfig): Database connection configuration
        PORT (int): Server port number
        SECRET_KEY (str): Application secret key
        API_KEY (str): API authentication key
        ALLOWED_HOSTS (list[str]): List of allowed host addresses
        MODEL_ID (str): Identifier for the ML model
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

    PORT: int = Field(default=8000)

    SECRET_KEY: str = Field(default=os.environ["SECRET_KEY"])
    API_KEY: str = Field(default=os.environ["API_KEY"])
    ALLOWED_HOSTS: list[str] = Field(default=["localhost", "127.0.0.1"])

    MODEL_ID: str = Field(default="runwayml/stable-diffusion-v1-5")
    MAX_TOKENS: int = Field(default=1000)

    class Config:
        """Pydantic configuration class.

        Attributes:
            env_file: Path to .env file
            env_file_encoding: Encoding for .env file
            case_sensitive: Case sensitivity flag for fields
        """

        env_file = os.path.join(basedir, ".env")
        env_file_encoding = "utf-8"
        case_sensitive = True


class AppConfig:
    """Singleton configuration class for application-wide settings.

    This class implements the singleton pattern to ensure only one instance
    of the configuration is created and used throughout the application.

    The singleton pattern ensures consistent configuration access across
    the entire application lifetime.

    Attributes:
        _instance: Internal singleton instance reference
        _settings: Internal settings instance

    Example:
        # Get configuration instance
        config = AppConfig()

        # Access settings
        redis_url = config.redis.url
        is_prod = config.is_production()
    """

    _instance = None
    _settings: Optional[Settings] = None

    def __new__(cls):
        """Create or return the singleton instance.

        Returns:
            AppConfig: The singleton configuration instance
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._settings = Settings()
        return cls._instance

    @property
    def settings(self) -> Settings:
        """Get the settings instance.

        Returns:
            Settings: The application settings instance
        """
        if not self._settings:
            self._settings = Settings()
        return self._settings

    @property
    def redis(self) -> RedisConfig:
        """Get Redis configuration.

        Returns:
            RedisConfig: Redis connection settings
        """
        return self.settings.REDIS

    @property
    def modal(self) -> ModalConfig:
        """Get Modal configuration.

        Returns:
            ModalConfig: Modal compute settings
        """
        return self.settings.MODAL

    @property
    def database(self) -> DatabaseConfig:
        """Get database configuration.

        Returns:
            DatabaseConfig: Database connection settings
        """
        return self.settings.DATABASE

    @property
    def port(self) -> int:
        """Get port configuration.

        Returns:
            int: Server port number
        """
        return self.settings.PORT

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
