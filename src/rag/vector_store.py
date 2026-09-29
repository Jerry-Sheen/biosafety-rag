"""
Persistent ChromaDB Vector Store Manager.
Maintains persistent on-disk vector database, supports incremental document additions,
metadata filtering, statistics tracking, and document deletion.
"""
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from src.rag.embeddings import OllamaEmbeddingProvider
from src.security.logger import get_secure_logger

logger = get_secure_logger("vector_store")


class PersistentVectorStore:
    """Manages persistent ChromaDB vector storage with incremental append support."""

    def __init__(
        self,
        persist_directory: Optional[Path] = None,
        collection_name: Optional[str] = None,
        embedding_provider: Optional[OllamaEmbeddingProvider] = None
    ):
        self.persist_directory = str(persist_directory or config.chroma_persist_dir)
        self.collection_name = collection_name or config.chroma_collection_name
        self.embedding_provider = embedding_provider or OllamaEmbeddingProvider()

        logger.info(f"Connecting to ChromaDB at: {self.persist_directory}")
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Incrementally appends chunked documents to the persistent collection.
        Does NOT overwrite existing data; appends seamlessly.
        """
        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]

        # Generate unique deterministic/UUID IDs
        ids = [
            f"{m.get('source', 'doc')}_{m.get('chunk_index', idx)}_{uuid.uuid4().hex[:8]}"
            for idx, m in enumerate(metadatas)
        ]

        logger.info(f"Generating embeddings for {len(texts)} document chunks...")
        embeddings = self.embedding_provider.embed_documents(texts)

        # Chroma metadata requires values to be string, int, float, or bool
        cleaned_metadatas = []
        for meta in metadatas:
            cleaned = {}
            for k, v in meta.items():
                if isinstance(v, (str, int, float, bool)):
                    cleaned[k] = v
                else:
                    cleaned[k] = str(v)
            cleaned_metadatas.append(cleaned)

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=cleaned_metadatas
        )

        logger.info(f"Successfully added {len(chunks)} chunks to collection '{self.collection_name}'.")
        return len(chunks)

    def search(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Searches the collection for chunks relevant to the user query.
        Returns sorted list of matching chunks with similarity scores.
        """
        if self.collection.count() == 0:
            logger.info("Search requested but collection is empty.")
            return []

        query_embedding = self.embedding_provider.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        matched_items = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, distances):
            # For cosine distance: distance is 1 - cosine_similarity (range 0 to 2)
            # Similarity = 1 - distance
            similarity = max(0.0, 1.0 - dist)
            if similarity >= score_threshold:
                matched_items.append({
                    "text": doc,
                    "metadata": meta,
                    "similarity": round(similarity, 4),
                    "distance": round(dist, 4)
                })

        # Sort descending by similarity
        matched_items.sort(key=lambda x: x["similarity"], reverse=True)
        return matched_items

    def get_stats(self) -> Dict[str, Any]:
        """Returns collection summary statistics."""
        count = self.collection.count()
        if count == 0:
            return {
                "total_chunks": 0,
                "unique_documents_count": 0,
                "document_sources": []
            }

        # Retrieve all metadatas to aggregate distinct document sources
        all_meta = self.collection.get(include=["metadatas"]).get("metadatas", [])
        sources = set()
        source_counts: Dict[str, int] = {}
        for m in all_meta:
            src = m.get("source", "Unknown")
            sources.add(src)
            source_counts[src] = source_counts.get(src, 0) + 1

        return {
            "total_chunks": count,
            "unique_documents_count": len(sources),
            "document_sources": sorted(list(sources)),
            "source_chunk_counts": source_counts
        }

    def delete_document(self, source_filename: str) -> int:
        """Deletes all chunks belonging to a specific source file."""
        logger.info(f"Deleting all chunks for document source: {source_filename}")
        initial_count = self.collection.count()
        self.collection.delete(where={"source": source_filename})
        new_count = self.collection.count()
        deleted = initial_count - new_count
        logger.info(f"Deleted {deleted} chunks for '{source_filename}'.")
        return deleted

    def reset_collection(self):
        """Clears all records in the collection."""
        logger.warning(f"Resetting vector store collection '{self.collection_name}'...")
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
