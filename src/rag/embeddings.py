"""
Local Embedding Provider Module via Ollama.
Utilizes Ollama with high-performance Korean & multilingual embedding models (default: bge-m3).
Completely free, runs 100% on local hardware, no external API keys required.
Includes resilient mock fallback for offline test environments.
"""
from typing import List, Optional
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

    def embed_documents(self, documents: List[str], batch_size: int = 16) -> List[List[float]]:
        """Embeds a list of document strings with batching."""
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
                logger.error(f"Ollama batch embedding failed for [{i}:{i+len(batch)}]: {e}. Using fallback.")
                for doc in batch:
                    all_embeddings.append(self.embed_query(doc))

        return all_embeddings
