"""Retrieval helpers for lexical, vector, and RRF hybrid search."""

from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

from rank_bm25 import BM25Okapi

try:
    from src.rag.config import (
        RAG_EMBEDDING_PROVIDER,
        RAG_CHROMA_PATH,
        RAG_COLLECTION_NAME,
        RAG_INGEST_VERSION,
    )
    from src.rag.generation.cite_answer import write_query_results_docx
    from src.rag.ingestion.embed_store import _embed_texts, _get_chroma_client
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from src.rag.config import (
        RAG_EMBEDDING_PROVIDER,
        RAG_CHROMA_PATH,
        RAG_COLLECTION_NAME,
        RAG_INGEST_VERSION,
    )
    from src.rag.generation.cite_answer import write_query_results_docx
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


def reciprocal_rank_fusion(
    bm25_results: list[dict[str, Any]],
    vector_results: list[dict[str, Any]],
    top_k: int = 3,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    fused_scores: dict[str, float] = defaultdict(float)
    fused_records: dict[str, dict[str, Any]] = {}

    for rank, result in enumerate(bm25_results, start=1):
        chunk_id = result["id"]
        fused_scores[chunk_id] += 1.0 / (rrf_k + rank)
        fused_records.setdefault(
            chunk_id,
            {
                "id": chunk_id,
                "document": result["document"],
                "metadata": result["metadata"],
                "matched_by": set(),
            },
        )
        fused_records[chunk_id]["bm25_score"] = result["bm25_score"]
        fused_records[chunk_id]["bm25_rank"] = rank
        fused_records[chunk_id]["matched_by"].add("bm25")

    for rank, result in enumerate(vector_results, start=1):
        chunk_id = result["id"]
        fused_scores[chunk_id] += 1.0 / (rrf_k + rank)
        fused_records.setdefault(
            chunk_id,
            {
                "id": chunk_id,
                "document": result["document"],
                "metadata": result["metadata"],
                "matched_by": set(),
            },
        )
        fused_records[chunk_id]["distance"] = result["distance"]
        fused_records[chunk_id]["vector_rank"] = rank
        fused_records[chunk_id]["vector_provider"] = result.get("vector_provider")
        fused_records[chunk_id]["matched_by"].add("vector")

    fused_results: list[dict[str, Any]] = []
    for chunk_id, record in fused_records.items():
        fused_result = dict(record)
        fused_result["rrf_score"] = fused_scores[chunk_id]
        fused_result["matched_by"] = "+".join(sorted(record["matched_by"]))
        fused_results.append(fused_result)

    fused_results.sort(key=lambda item: item["rrf_score"], reverse=True)
    return fused_results[:top_k]


def hybrid_search(
    query: str,
    top_k: int = 3,
    candidate_k: int | None = None,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    candidate_count = candidate_k or max(top_k, 3)
    bm25_results = bm25_search(query, top_k=candidate_count)
    vector_results = vector_search(query, top_k=candidate_count)
    return reciprocal_rank_fusion(
        bm25_results=bm25_results,
        vector_results=vector_results,
        top_k=top_k,
        rrf_k=rrf_k,
    )


def check_query(output_path: str = "artifacts/hybrid_search_report.docx") -> None:
    reports: list[dict[str, Any]] = []
    for query in DEFAULT_QUERIES:
        print(f"Query: {query}")
        results = hybrid_search(query, top_k=3)
        reports.append({"query": query, "results": results})

        for idx, result in enumerate(results, start=1):
            print(f"Result {idx}:")
            print(f"Document: {result['document']}")
            print(f"Metadata: {result['metadata']}")
            print(f"RRF Score: {result['rrf_score']}")
            print(f"BM25 Score: {result.get('bm25_score', 'N/A')}")
            print(f"Vector Distance: {result.get('distance', 'N/A')}\n")

    saved_path = write_query_results_docx(output_path=output_path, reports=reports, title="Hybrid Search Report")
    print(f"Saved Word report to: {saved_path}")


if __name__ == "__main__":
    check_query()
