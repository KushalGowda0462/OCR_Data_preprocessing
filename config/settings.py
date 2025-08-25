# config/settings.py
import os
from typing import Optional
from pydantic import BaseSettings  # Use pydantic instead of pydantic_settings

class Settings(BaseSettings):
    # Redis settings
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    redis_db: int = int(os.getenv("REDIS_DB", "0"))
    
    # S3/MinIO settings
    s3_endpoint: Optional[str] = os.getenv("S3_ENDPOINT")
    s3_bucket: str = os.getenv("S3_BUCKET", "document-processing")
    s3_access_key: Optional[str] = os.getenv("S3_ACCESS_KEY")
    s3_secret_key: Optional[str] = os.getenv("S3_SECRET_KEY")
    
    # Application settings
    app_host: str = os.getenv("APP_HOST", "0.0.0.0")
    app_port: int = int(os.getenv("APP_PORT", "8000"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Preprocessing settings
    deskew_enabled: bool = os.getenv("DESKEW_ENABLED", "True").lower() == "true"
    line_removal_enabled: bool = os.getenv("LINE_REMOVAL_ENABLED", "True").lower() == "true"

    class Config:
        env_file = ".env"

settings = Settings()