"""
Application configuration management.
Centralizes paths, Ollama settings, models, and system parameters.
"""
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from src.security.secrets_manager import secrets_manager

BASE_DIR = Path(__file__).resolve().parent.parent

class AppConfig(BaseModel):
    """Application configuration schema."""
    app_env: str = Field(default_factory=lambda: secrets_manager.get_secret("APP_ENV", "development"))
    app_host: str = Field(default_factory=lambda: secrets_manager.get_secret("APP_HOST", "0.0.0.0"))
    admin_port: int = Field(default_factory=lambda: int(secrets_manager.get_secret("ADMIN_PORT", "8501")))
    user_port: int = Field(default_factory=lambda: int(secrets_manager.get_secret("USER_PORT", "8502")))

    # LLM Provider: "gemini" (Cloud API, no local Ollama required) or "ollama" (Local server)
    llm_provider: str = Field(
        default_factory=lambda: secrets_manager.get_secret("LLM_PROVIDER", "gemini")
    )

    # Google Gemini Cloud API Configuration (Free tier, fast streaming, highly scalable for multi-user)
    gemini_api_key: str = Field(
        default_factory=lambda: secrets_manager.get_secret("GEMINI_API_KEY", "")
    )
    gemini_model: str = Field(
        default_factory=lambda: secrets_manager.get_secret("GEMINI_MODEL", "gemini-3.8-flash")
    )
    gemini_embed_model: str = Field(
        default_factory=lambda: secrets_manager.get_secret("GEMINI_EMBED_MODEL", "text-embedding-004")
    )

    # Local Ollama Configuration (Optional fallback when running on local machine)
    ollama_base_url: str = Field(
        default_factory=lambda: secrets_manager.get_secret("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    # LLM Model: exaone3.5 (LG AI Research Korean SOTA) or qwen2.5:7b / llama3.1
    ollama_llm_model: str = Field(
        default_factory=lambda: secrets_manager.get_secret("OLLAMA_LLM_MODEL", "exaone3.5")
    )
    # Embedding Model: bge-m3 (SOTA multilingual & Korean embedding) or nomic-embed-text
    ollama_embed_model: str = Field(
        default_factory=lambda: secrets_manager.get_secret("OLLAMA_EMBED_MODEL", "bge-m3")
    )

    # Vector Store & Storage Paths
    chroma_persist_dir: Path = Field(
        default_factory=lambda: BASE_DIR / secrets_manager.get_secret("CHROMA_PERSIST_DIRECTORY", "data/chroma_db")
    )
    chroma_collection_name: str = Field(
        default_factory=lambda: secrets_manager.get_secret("CHROMA_COLLECTION_NAME", "rag_documents")
    )
    upload_dir: Path = Field(
        default_factory=lambda: BASE_DIR / secrets_manager.get_secret("UPLOAD_DIRECTORY", "data/uploads")
    )

    # Security & Admin Access PIN
    admin_access_pin: str = Field(
        default_factory=lambda: secrets_manager.get_secret("ADMIN_ACCESS_PIN", "admin1234")
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def init_directories(self):
        """Ensures storage directories exist."""
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)


config = AppConfig()
config.init_directories()
