"""Retrieval helpers for lexical and vector candidate collection."""

from pathlib import Path
import sys
from typing import Any

from rank_bm25 import BM25Okapi

try:
    from src.rag.config import (
        RAG_CANDIDATE_K,
        RAG_CHROMA_PATH,
        RAG_COLLECTION_NAME,
        RAG_EMBEDDING_PROVIDER,
        RAG_INGEST_VERSION,
    )
    from src.rag.ingestion.embed_store import _embed_texts, _get_chroma_client
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from src.rag.config import (
        RAG_CANDIDATE_K,
        RAG_CHROMA_PATH,
        RAG_COLLECTION_NAME,
        RAG_EMBEDDING_PROVIDER,
        RAG_INGEST_VERSION,
    )
    from src.rag.ingestion.embed_store import _embed_texts, _get_chroma_client


DEFAULT_QUERIES = [
    "What is the transformer architecture?",
    "How is self-attention computed?",
    "What are the main contributions of this paper?",
]


def _tokenize_for_bm25(text: str) -> list[str]:
    return text.lower().split()


def _get_collection():
    client, _storage_mode = _get_chroma_client(chroma_path=RAG_CHROMA_PATH)
    return client.get_or_create_collection(name=RAG_COLLECTION_NAME)


def _load_chunk_corpus_from_chroma() -> list[dict[str, Any]]:
    collection = _get_collection()
    rows = collection.get(
        where={"ingest_version": RAG_INGEST_VERSION},
        include=["documents", "metadatas"],
    )

    ids = rows.get("ids", [])
    documents = rows.get("documents", [])
    metadatas = rows.get("metadatas", [])

    chunk_records: list[dict[str, Any]] = []
    for chunk_id, document, metadata in zip(ids, documents, metadatas):
        if not document:
            continue
        chunk_records.append(
            {
                "id": chunk_id,
                "document": document,
                "metadata": metadata or {},
            }
        )
    return chunk_records


def bm25_search(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    chunk_records = _load_chunk_corpus_from_chroma()
    if not chunk_records:
        return []

    tokenized_corpus = [
        _tokenize_for_bm25(record["document"]) for record in chunk_records
    ]
    bm25 = BM25Okapi(tokenized_corpus)
    query_tokens = _tokenize_for_bm25(query)
    scores = bm25.get_scores(query_tokens)

    scored_results: list[dict[str, Any]] = []
    for record, score in zip(chunk_records, scores):
        scored_results.append(
            {
                "id": record["id"],
                "document": record["document"],
                "metadata": record["metadata"],
                "bm25_score": float(score),
            }
        )

    scored_results.sort(key=lambda item: item["bm25_score"], reverse=True)
    return scored_results[:top_k]


def vector_search(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    collection = _get_collection()
    query_embeddings, provider_used = _embed_texts(
        [query], provider=RAG_EMBEDDING_PROVIDER
    )
    results = collection.query(
        query_embeddings=query_embeddings,
        n_results=top_k,
        where={"ingest_version": RAG_INGEST_VERSION},
        include=["documents", "metadatas", "distances"],
    )

    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    scored_results: list[dict[str, Any]] = []
    for chunk_id, document, metadata, distance in zip(
        ids, documents, metadatas, distances
    ):
        if not document:
            continue
        scored_results.append(
            {
                "id": chunk_id,
                "document": document,
                "metadata": metadata or {},
                "distance": float(distance),
                "vector_provider": provider_used,
            }
        )

    scored_results.sort(key=lambda item: item["distance"])
    return scored_results[:top_k]


def combine_search_results(
    bm25_results: list[dict[str, Any]],
    vector_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge BM25 and vector top chunks into a deduplicated rerank candidate list."""
    combined_records: dict[str, dict[str, Any]] = {}

    for rank, result in enumerate(bm25_results, start=1):
        chunk_id = result["id"]
        combined_records.setdefault(
            chunk_id,
            {
                "id": chunk_id,
                "document": result["document"],
                "metadata": result["metadata"],
                "matched_by": set(),
            },
        )
        combined_records[chunk_id]["bm25_score"] = result["bm25_score"]
        combined_records[chunk_id]["bm25_rank"] = rank
        combined_records[chunk_id]["matched_by"].add("bm25")

    for rank, result in enumerate(vector_results, start=1):
        chunk_id = result["id"]
        combined_records.setdefault(
            chunk_id,
            {
                "id": chunk_id,
                "document": result["document"],
                "metadata": result["metadata"],
                "matched_by": set(),
            },
        )
        combined_records[chunk_id]["distance"] = result["distance"]
        combined_records[chunk_id]["vector_rank"] = rank
        combined_records[chunk_id]["matched_by"].add("vector")

    combined_results: list[dict[str, Any]] = []
    for record in combined_records.values():
        combined_result = dict(record)
        combined_result["matched_by"] = "+".join(sorted(record["matched_by"]))
        combined_results.append(combined_result)

    combined_results.sort(
        key=lambda item: (
            item.get("bm25_rank", float("inf")),
            item.get("vector_rank", float("inf")),
            item["id"],
        )
    )
    return combined_results


def collect_rerank_candidates(
    query: str,
    top_k: int = RAG_CANDIDATE_K,
    candidate_k: int | None = None,
) -> list[dict[str, Any]]:
    """Return top chunks from BM25 and vector search for Cohere reranking."""
    candidate_count = candidate_k or top_k
    bm25_results = bm25_search(query, top_k=candidate_count)
    vector_results = vector_search(query, top_k=candidate_count)
    return combine_search_results(bm25_results=bm25_results, vector_results=vector_results)


if __name__ == "__main__":
    pass
