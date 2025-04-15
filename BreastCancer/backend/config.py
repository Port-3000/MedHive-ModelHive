# app/config.py
import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    """Application settings."""
    
    # Application settings
    APP_NAME: str = "Breast Cancer Prediction API"
    APP_VERSION: str = "1.0.0"
    API_PREFIX: str = ""
    
    # Model settings
    MODEL_PATH: str = os.environ.get("MODEL_PATH", "../models/logistic_regression_model.pkl")
    
    # Hugging Face Spaces settings
    HF_SPACE_ID: str = os.environ.get("HF_SPACE_ID", "")
    
    # Logging settings
    LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create settings instance
settings = Settings()