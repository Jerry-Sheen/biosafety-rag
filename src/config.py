"""
Application configuration management.
Centralizes paths, model names, and system parameters.
"""
import os
from pathlib import Path
from pydantic import BaseModel, Field
from src.security.secrets_manager import secrets_manager

BASE_DIR = Path(__file__).resolve().parent.parent

class AppConfig(BaseModel):
    """Application configuration schema."""
    app_env: str = Field(default_factory=lambda: secrets_manager.get_secret("APP_ENV", "development"))
    app_host: str = Field(default_factory=lambda: secrets_manager.get_secret("APP_HOST", "0.0.0.0"))
    app_port: int = Field(default_factory=lambda: int(secrets_manager.get_secret("APP_PORT", "8501")))

    # LLM & Embedding Models
    gemini_api_key: str = Field(default_factory=lambda: secrets_manager.get_secret("GEMINI_API_KEY", ""))
    llM_model: str = Field(default_factory=lambda: secrets_manager.get_secret("LLM_MODEL", "gemini-2.5-flash"))
    embedding_model: str = Field(default_factory=lambda: secrets_manager.get_secret("EMBEDDING_MODEL", "text-embedding-004"))

    # Vector Store & Storage
    chroma_persist_dir: Path = Field(
        default_factory=lambda: BASE_DIR / secrets_manager.get_secret("CHROMA_PERSIST_DIRECTORY", "data/chroma_db")
    )
    chroma_collection_name: str = Field(
        default_factory=lambda: secrets_manager.get_secret("CHROMA_COLLECTION_NAME", "rag_documents")
    )
    upload_dir: Path = Field(
        default_factory=lambda: BASE_DIR / secrets_manager.get_secret("UPLOAD_DIRECTORY", "data/uploads")
    )

    # Security & Admin Access
    admin_access_pin: str = Field(
        default_factory=lambda: secrets_manager.get_secret("ADMIN_ACCESS_PIN", "admin1234")
    )

    class Config:
        arbitrary_types_allowed = True

    def init_directories(self):
        """Ensures storage directories exist."""
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)


config = AppConfig()
config.init_directories()
