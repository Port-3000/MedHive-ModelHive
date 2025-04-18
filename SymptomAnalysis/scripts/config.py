
# Configuration management for Disease Diagnosis System
# src/config.py

import os
from pydantic import BaseSettings, Field
from typing import List, Optional
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings, loaded from environment variables"""
    
    # App information
    APP_NAME: str = "Disease Diagnosis System"
    APP_DESCRIPTION: str = "An AI-powered system for diagnosing diseases based on symptoms"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=False, env="DEBUG")
    
    # Server settings
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    
    # CORS settings
    CORS_ORIGINS: List[str] = Field(default=["*"])
    
    # AstraDB settings
    ASTRA_DB_ID: str = Field(..., env="ASTRA_DB_ID")
    ASTRA_DB_REGION: str = Field(..., env="ASTRA_DB_REGION")
    ASTRA_DB_APPLICATION_TOKEN: str = Field(..., env="ASTRA_DB_APPLICATION_TOKEN")
    ASTRA_DB_KEYSPACE: str = Field(default="disease_diagnosis", env="ASTRA_DB_KEYSPACE")
    ASTRA_SECURE_BUNDLE_PATH: str = Field(default="secure-connect-disease-diagnosis.zip", env="ASTRA_SECURE_BUNDLE_PATH")
    
    # Groq API settings
    GROQ_API_KEY: str = Field(..., env="GROQ_API_KEY")
    GROQ_MODEL_NAME: str = Field(default="mixtral-8x7b-32768", env="GROQ_MODEL_NAME")
    GROQ_SYMPTOM_MODEL: str = Field(default="llama3-70b-8192", env="GROQ_SYMPTOM_MODEL")
    
    # Embedding model settings
    EMBEDDING_MODEL: str = Field(default="all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    
    # Logging settings
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", env="LOG_FORMAT")
    
    # ML Flow settings (optional, for future use)
    MLFLOW_TRACKING_URI: Optional[str] = Field(default=None, env="MLFLOW_TRACKING_URI")
    
    # Supabase settings (optional, for future use)
    SUPABASE_URL: Optional[str] = Field(default=None, env="SUPABASE_URL")
    SUPABASE_KEY: Optional[str] = Field(default=None, env="SUPABASE_KEY")
    
    class Config:
        """Configuration settings"""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

# Export settings instance
settings = get_settings()