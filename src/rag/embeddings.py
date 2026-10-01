"""
Local Embedding Provider Module via Ollama.
Utilizes Ollama with high-performance Korean & multilingual embedding models (default: bge-m3).
Completely free, runs 100% on local hardware, no external API keys required.
Includes resilient mock fallback for offline test environments.
"""
from typing import List, Optional, Any
import numpy as np
from src.config import config
from src.rag.ollama_client import OllamaClient
from src.security.logger import get_secure_logger

logger = get_secure_logger("embeddings")


class OllamaEmbeddingProvider:
    """Manages document and query vector embeddings using local Ollama instance."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        self.base_url = base_url or config.ollama_base_url
        self.model_name = model_name or config.ollama_embed_model
        self.client = OllamaClient(base_url=self.base_url)
        self._is_mock = False

        # Check connection on initialization
        status = self.client.check_connection()
        if not status["connected"]:
            logger.warning(
                f"Cannot connect to Ollama at {self.base_url} ({status['error']}). "
                "Running in MOCK embedding mode until Ollama is started."
            )
            self._is_mock = True
        else:
            installed = status.get("models", [])
            model_installed = any(
                m == self.model_name or m.startswith(f"{self.model_name}:")
                for m in installed
            )
            if not model_installed:
                logger.warning(
                    f"Ollama connected, but embedding model '{self.model_name}' is not installed yet. "
                    f"Run 'ollama pull {self.model_name}' in your terminal. "
                    "Running in MOCK embedding mode in the meantime."
                )
                self._is_mock = True
            else:
                logger.info(f"Connected to Ollama at {self.base_url}. Using embedding model: {self.model_name}")

    @property
    def is_mock(self) -> bool:
        return self._is_mock

    def _generate_mock_embedding(self, text: str, dimension: int = 1024) -> List[float]:
        """Generates deterministic unit vector for testing/offline scenarios."""
        seed = abs(hash(text)) % (2**32)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(dimension).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Embeds a single user query string."""
        if self._is_mock:
            return self._generate_mock_embedding(query)

        try:
            embeddings = self.client.embed(model=self.model_name, texts=[query])
            if embeddings and len(embeddings) > 0:
                return embeddings[0]
            return self._generate_mock_embedding(query)
        except Exception as e:
            logger.error(f"Error calling Ollama embed for query: {e}. Falling back to deterministic vector.")
            return self._generate_mock_embedding(query)

    def embed_documents(self, documents: List[str], batch_size: int = 4) -> List[List[float]]:
        """Embeds a list of document strings with small batch size for local stability."""
        if not documents:
            return []

        if self._is_mock:
            return [self._generate_mock_embedding(doc) for doc in documents]

        all_embeddings: List[List[float]] = []

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            try:
                embeddings = self.client.embed(model=self.model_name, texts=batch)
                all_embeddings.extend(embeddings)
            except Exception as e:
                logger.error(f"Ollama batch embedding failed for [{i}:{i+len(batch)}]: {e}. Trying individual items.")
                for doc in batch:
                    all_embeddings.append(self.embed_query(doc))

        return all_embeddings


class GeminiEmbeddingProvider:
    """Manages document and query vector embeddings using Google Gemini Cloud API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        self.api_key = api_key or config.gemini_api_key
        self.model_name = model_name or config.gemini_embed_model

    def embed_query(self, query: str) -> List[float]:
        """Embeds a single query string using Gemini."""
        if not self.api_key:
            logger.warning("Gemini API key not found. Using fallback embedding vector.")
            return [0.0] * 768

        try:
            from google import genai
            client = genai.Client(api_key=self.api_key)
            result = client.models.embed_content(
                model=self.model_name,
                contents=query
            )
            if hasattr(result, "embedding") and hasattr(result.embedding, "values"):
                return result.embedding.values
            elif hasattr(result, "embeddings") and len(result.embeddings) > 0:
                return result.embeddings[0].values
            return [0.0] * 768
        except Exception as e:
            logger.error(f"Gemini embed_query error: {e}")
            return [0.0] * 768

    def embed_documents(self, documents: List[str], batch_size: int = 8) -> List[List[float]]:
        """Embeds multiple document chunks with Gemini."""
        if not documents:
            return []
        if not self.api_key:
            return [[0.0] * 768 for _ in documents]

        all_embeddings: List[List[float]] = []
        try:
            from google import genai
            client = genai.Client(api_key=self.api_key)

            for i in range(0, len(documents), batch_size):
                batch = documents[i : i + batch_size]
                result = client.models.embed_content(
                    model=self.model_name,
                    contents=batch
                )
                if hasattr(result, "embeddings"):
                    all_embeddings.extend([emb.values for emb in result.embeddings])
                else:
                    for doc in batch:
                        all_embeddings.append(self.embed_query(doc))
            return all_embeddings
        except Exception as e:
            logger.error(f"Gemini embed_documents error: {e}")
            for doc in documents:
                all_embeddings.append(self.embed_query(doc))
            return all_embeddings


def get_embedding_provider() -> Any:
    """Returns the most appropriate embedding provider based on available environment."""
    ollama_provider = OllamaEmbeddingProvider()
    if not ollama_provider.is_mock:
        return ollama_provider
    elif config.gemini_api_key:
        logger.info("Using Gemini Cloud Embedding Provider.")
        return GeminiEmbeddingProvider()
    return ollama_provider

