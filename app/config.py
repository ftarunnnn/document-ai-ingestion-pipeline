"""
Application Configuration Settings.
Loads configuration from environment variables or uses production-ready defaults.
"""

from pathlib import Path
from typing import List, Set
from pydantic import Field
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseModel as BaseSettings  # type: ignore

# Base project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
DB_DIR = DATA_DIR / "db"

# Ensure runtime directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    """Pipeline and service configuration parameters."""

    # Application Info
    APP_NAME: str = "Document AI Ingestion Pipeline"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    API_PREFIX: str = "/api/v1"

    # Storage Paths
    UPLOAD_FOLDER: Path = UPLOAD_DIR
    PROCESSED_FOLDER: Path = PROCESSED_DIR
    DB_FOLDER: Path = DB_DIR

    # Validation Rules
    MAX_FILE_SIZE_MB: int = 50
    ALLOWED_EXTENSIONS: Set[str] = {
        ".pdf",
        ".docx",
        ".doc",
        ".xlsx",
        ".xls",
        ".txt",
        ".png",
        ".jpg",
        ".jpeg",
        ".tiff",
        ".bmp",
        ".webp",
    }

    ALLOWED_MIME_TYPES: Set[str] = {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel",
        "text/plain",
        "image/png",
        "image/jpeg",
        "image/tiff",
        "image/bmp",
        "image/webp",
    }

    # Database Settings
    # Supports SQLite by default or PostgreSQL + pgvector
    DATABASE_URL: str = f"sqlite:///{DB_DIR / 'pipeline.db'}"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "document_ai"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"

    # Vector & Embeddings
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150

    # Task Queue (Celery / Redis)
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"

    # OCR Settings
    TESSERACT_CMD: str = "tesseract"

    # API Security
    API_KEY_ENABLED: bool = False
    API_KEY: str = "docai-secret-key-production"

    model_config = {
        "env_file": ".env",
        "extra": "ignore",
    }


settings = Settings()
