"""
Configuration management for Ocean ML platform
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings"""
    
    # Database
    database_url: str = "postgresql://user:password@localhost:5432/oceanml"
    redis_url: str = "redis://localhost:6379/0"
    
    # Storage
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    s3_bucket_name: str = "ocean-ml-data"
    
    # LLM APIs (Optional)
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    enable_llm_tagging: bool = False
    
    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    wandb_api_key: Optional[str] = None
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = True
    log_level: str = "INFO"
    
    # Training
    max_training_jobs: int = 5
    default_batch_size: int = 32
    default_epochs: int = 50
    
    # Paths
    data_dir: str = "./data"
    models_dir: str = "./data/models"
    raw_data_dir: str = "./data/raw"
    processed_data_dir: str = "./data/processed"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
