"""
Unit tests for Persistent ChromaDB Vector Store.
"""
import tempfile
from pathlib import Path
from src.rag.vector_store import PersistentVectorStore
from src.rag.embeddings import GeminiEmbeddingProvider


def test_vector_store_incremental_add_and_search():
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Use mock embedding provider for tests
        embedder = GeminiEmbeddingProvider(api_key="mock_test_key")
        vs = PersistentVectorStore(
            persist_directory=Path(tmp_dir),
            collection_name="test_collection",
            embedding_provider=embedder
        )

        # Batch 1
        chunks_doc1 = [
            {"text": "Python은 데이터 과학과 인공지능 개발에 널리 사용되는 언어입니다.", "metadata": {"source": "python.pdf", "chunk_index": 0}},
            {"text": "Streamlit은 손쉽게 웹 대시보드를 구축할 수 있는 프레임워크입니다.", "metadata": {"source": "python.pdf", "chunk_index": 1}},
        ]
        added_1 = vs.add_documents(chunks_doc1)
        assert added_1 == 2
        assert vs.get_stats()["total_chunks"] == 2

        # Batch 2: Incremental addition (different document)
        chunks_doc2 = [
            {"text": "Docker는 컨테이너 기술을 활용하여 애플리케이션을 격리 배포합니다.", "metadata": {"source": "docker.pdf", "chunk_index": 0}},
        ]
        added_2 = vs.add_documents(chunks_doc2)
        assert added_2 == 1
        # Check that previous data was NOT overwritten
        stats = vs.get_stats()
        assert stats["total_chunks"] == 3
        assert stats["unique_documents_count"] == 2
        assert "python.pdf" in stats["document_sources"]
        assert "docker.pdf" in stats["document_sources"]

        # Search
        results = vs.search(query="Python", top_k=2)
        assert len(results) > 0
        assert "similarity" in results[0]

        # Delete document
        deleted = vs.delete_document("python.pdf")
        assert deleted == 2
        assert vs.get_stats()["total_chunks"] == 1
        assert "python.pdf" not in vs.get_stats()["document_sources"]
