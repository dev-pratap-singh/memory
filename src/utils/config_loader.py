"""
Configuration Loader
Loads and validates configuration from YAML and environment variables
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


def resolve_env_vars(value: Any) -> Any:
    """Resolve environment variables in configuration values"""
    if isinstance(value, str):
        # Pattern: ${VAR_NAME:default_value}
        pattern = r'\$\{([^:}]+)(?::([^}]+))?\}'
        matches = re.findall(pattern, value)

        for var_name, default in matches:
            env_value = os.getenv(var_name, default)
            value = value.replace(f'${{{var_name}:{default}}}', env_value)
            value = value.replace(f'${{{var_name}}}', env_value)

    elif isinstance(value, dict):
        return {k: resolve_env_vars(v) for k, v in value.items()}

    elif isinstance(value, list):
        return [resolve_env_vars(item) for item in value]

    return value


class DatabaseConfig(BaseModel):
    """Database configuration"""
    host: str = "localhost"
    port: int = 5432
    name: str = "memory_db"
    user: str = "postgres"
    password: str = "postgres"
    pool_size: int = 10
    max_overflow: int = 20

    @property
    def url(self) -> str:
        """Get database URL"""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class SLMConfig(BaseModel):
    """Small Language Model configuration"""
    current: Dict[str, Any]
    next: Dict[str, Any]
    lora: Dict[str, Any]
    training: Dict[str, Any]
    retraining: Dict[str, Any]


class EmbeddingConfig(BaseModel):
    """Embedding model configuration"""
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimension: int = 384
    batch_size: int = 32
    cache_dir: str = "./data/embeddings_cache"


class PrimaryLLMConfig(BaseModel):
    """Primary LLM configuration"""
    provider: str = "openai"
    model: str = "gpt-4-turbo-preview"
    temperature: float = 0.7
    max_tokens: int = 4096
    context_window: int = 128000
    fallback: Optional[Dict[str, str]] = None


class MemoryConfig(BaseModel):
    """Memory management configuration"""
    working_memory: Dict[str, Any]
    storage: Dict[str, Any]  # Changed from Dict[str, str] to Dict[str, Any]
    retention: Dict[str, Any]
    context_window: Dict[str, Any]


class APIConfig(BaseModel):
    """API server configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = False
    log_level: str = "info"
    cors_origins: list = ["*"]
    rate_limit: Dict[str, Any] = {}


class Config(BaseSettings):
    """Main application configuration"""
    database: DatabaseConfig
    slm: SLMConfig
    embeddings: EmbeddingConfig
    primary_llm: PrimaryLLMConfig
    memory: MemoryConfig
    tools: Dict[str, Any]
    api: APIConfig
    monitoring: Dict[str, Any]
    docker: Dict[str, Any]
    features: Dict[str, bool]
    performance: Dict[str, bool]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra fields from .env file


def load_config(config_path: str = "config.yaml") -> Config:
    """
    Load configuration from YAML file with environment variable resolution

    Args:
        config_path: Path to configuration file

    Returns:
        Loaded configuration object
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, 'r') as f:
        raw_config = yaml.safe_load(f)

    # Resolve environment variables
    resolved_config = resolve_env_vars(raw_config)

    # Create config object
    config = Config(**resolved_config)

    return config


def get_model_info(config: Config) -> Dict[str, Any]:
    """Get current and next model information"""
    return {
        "current_model": {
            "name": config.slm.current.get("name"),
            "context_window": config.slm.current.get("context_window"),
            "quantization": config.slm.current.get("quantization"),
        },
        "next_model": {
            "name": config.slm.next.get("name"),
            "context_window": config.slm.next.get("context_window"),
            "quantization": config.slm.next.get("quantization"),
        },
        "recommended_upgrade": config.slm.current.get("context_window", 2048) < 4096
    }


# Singleton instance
_config_instance: Optional[Config] = None


def get_config(reload: bool = False) -> Config:
    """
    Get configuration singleton

    Args:
        reload: Force reload configuration

    Returns:
        Configuration instance
    """
    global _config_instance

    if _config_instance is None or reload:
        _config_instance = load_config()

    return _config_instance


if __name__ == "__main__":
    # Test configuration loading
    from rich import print as rprint

    config = get_config()
    rprint("[green]Configuration loaded successfully![/green]")
    rprint(f"Database URL: {config.database.url}")
    rprint(f"SLM Model: {config.slm.current.get('name')}")
    rprint(f"Primary LLM: {config.primary_llm.model}")

    model_info = get_model_info(config)
    rprint("\n[yellow]Model Information:[/yellow]")
    rprint(model_info)
