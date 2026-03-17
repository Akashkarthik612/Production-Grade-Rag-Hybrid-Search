from pathlib import Path

from src.rag.ingestion.embed_store import ensure_local_papers_index
from src.rag.retrieval import retrieve as retrieve_module


def test_ensure_local_papers_index_reuses_when_documents_are_unchanged(tmp_path) -> None:
    papers_dir = tmp_path / "papers"
    chroma_path = tmp_path / "chroma"
    papers_dir.mkdir()
    (papers_dir / "doc1.txt").write_text(
        "Transformers use self-attention to process tokens in parallel.",
        encoding="utf-8",
    )

    first = ensure_local_papers_index(
        papers_dir=str(papers_dir),
        collection_name="test_collection",
        chroma_path=str(chroma_path),
        embedding_provider="hash",
        min_tokens=5,
        max_tokens=20,
        overlap=2,
        ingest_version="test-v1",
    )
    second = ensure_local_papers_index(
        papers_dir=str(papers_dir),
        collection_name="test_collection",
        chroma_path=str(chroma_path),
        embedding_provider="hash",
        min_tokens=5,
        max_tokens=20,
        overlap=2,
        ingest_version="test-v1",
    )

    assert first["action"] == "indexed"
    assert second["action"] == "reused"
    assert second["reason"] == "up_to_date"


def test_ensure_local_papers_index_rebuilds_and_warms_bm25_after_document_change(
    tmp_path,
    monkeypatch,
) -> None:
    papers_dir = tmp_path / "papers"
    chroma_path = tmp_path / "chroma"
    papers_dir.mkdir()
    document_path = papers_dir / "doc1.txt"
    document_path.write_text(
        "Transformers rely on self-attention and feed-forward layers.",
        encoding="utf-8",
    )

    ensure_local_papers_index(
        papers_dir=str(papers_dir),
        collection_name="test_collection",
        chroma_path=str(chroma_path),
        embedding_provider="hash",
        min_tokens=5,
        max_tokens=20,
        overlap=2,
        ingest_version="test-v1",
    )

    monkeypatch.setattr(retrieve_module, "RAG_CHROMA_PATH", str(chroma_path))
    monkeypatch.setattr(retrieve_module, "RAG_COLLECTION_NAME", "test_collection")
    monkeypatch.setattr(retrieve_module, "RAG_INGEST_VERSION", "test-v1")
    monkeypatch.setattr(retrieve_module, "RAG_EMBEDDING_PROVIDER", "hash")

    first_warm = retrieve_module.warm_retrieval_cache(force=True)
    initial_results = retrieve_module.bm25_search("transformers", top_k=1)

    document_path.write_text(
        "Reranking improves the final ordering of relevant retrieved chunks.",
        encoding="utf-8",
    )
    second = ensure_local_papers_index(
        papers_dir=str(papers_dir),
        collection_name="test_collection",
        chroma_path=str(chroma_path),
        embedding_provider="hash",
        min_tokens=5,
        max_tokens=20,
        overlap=2,
        ingest_version="test-v1",
    )

    retrieve_module.invalidate_retrieval_cache()
    second_warm = retrieve_module.warm_retrieval_cache(force=True)
    updated_results = retrieve_module.bm25_search("reranking", top_k=1)

    assert first_warm["bm25_ready"] is True
    assert initial_results
    assert second["action"] == "indexed"
    assert second["reason"] == "documents_changed"
    assert second_warm["bm25_ready"] is True
    assert updated_results
