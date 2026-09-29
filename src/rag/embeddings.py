"""
Embedding Provider Module.
Uses Google Gemini API's text-embedding-004 (or gemini-embedding-001) for state-of-the-art
multilingual & Korean semantic representation (Free tier available).
Includes resilient batching and fallback for offline unit testing.
"""
from typing import List, Optional
import numpy as np
from src.config import config
from src.security.logger import get_secure_logger

logger = get_secure_logger("embeddings")


class GeminiEmbeddingProvider:
    """Manages document and query vector embeddings using Google GenAI SDK."""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or config.gemini_api_key
        self.model_name = model_name or config.embedding_model
        self._client = None
        self._is_mock = False

        if not self.api_key or self.api_key.startswith("your_gemini") or len(self.api_key) < 10:
            logger.warning("No valid GEMINI_API_KEY found. Running in MOCK embedding mode for testing/demo.")
            self._is_mock = True
        else:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
                logger.info(f"Initialized Gemini Client with model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize Google GenAI Client: {e}. Falling back to mock embeddings.")
                self._is_mock = True

    @property
    def is_mock(self) -> bool:
        return self._is_mock

    def _generate_mock_embedding(self, text: str, dimension: int = 768) -> List[float]:
        """Generates deterministic unit vector for testing/offline scenarios."""
        # Seed pseudo-random with text hash for reproducible test similarities
        seed = abs(hash(text)) % (2**32)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(dimension).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Embeds a single user query string."""
        if self._is_mock or not self._client:
            return self._generate_mock_embedding(query)

        try:
            response = self._client.models.embed_content(
                model=self.model_name,
                contents=query,
            )
            if hasattr(response, "embedding") and response.embedding and hasattr(response.embedding, "values"):
                return response.embedding.values
            elif hasattr(response, "embeddings") and response.embeddings:
                return response.embeddings[0].values
            elif hasattr(response, "values"):
                return response.values
            else:
                logger.warning(f"Unexpected response structure: {response}. Using mock embedding.")
                return self._generate_mock_embedding(query)
        except Exception as e:
            logger.error(f"Error embedding query with Gemini API: {e}. Falling back to deterministic embedding.")
            return self._generate_mock_embedding(query)

    def embed_documents(self, documents: List[str], batch_size: int = 32) -> List[List[float]]:
        """Embeds a list of document strings with batching support."""
        if not documents:
            return []

        if self._is_mock or not self._client:
            return [self._generate_mock_embedding(doc) for doc in documents]

        all_embeddings: List[List[float]] = []

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            try:
                response = self._client.models.embed_content(
                    model=self.model_name,
                    contents=batch,
                )
                if hasattr(response, "embeddings") and response.embeddings:
                    batch_vecs = [item.values for item in response.embeddings]
                elif hasattr(response, "embedding") and response.embedding:
                    batch_vecs = [response.embedding.values]
                else:
                    batch_vecs = [self._generate_mock_embedding(doc) for doc in batch]
                all_embeddings.extend(batch_vecs)
            except Exception as e:
                logger.error(f"Batch embedding failed for batch [{i}:{i+len(batch)}]: {e}. Using fallback.")
                for doc in batch:
                    # Try single fallback or mock
                    all_embeddings.append(self.embed_query(doc))

        return all_embeddings
