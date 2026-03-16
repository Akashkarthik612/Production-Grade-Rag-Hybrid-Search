"""Optional Cohere reranking helpers for second-stage ranking."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

from cohere import ClientV2

try:
    from src.rag.config import (
        RAG_CANDIDATE_K,
        RAG_COHERE_API_KEY,
        RAG_COHERE_RERANK_MODEL,
        RAG_QUERY_TOP_K,
        RAG_RERANK_TOP_N,
    )
    from src.rag.retrieval.retrieve import collect_rerank_candidates
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from src.rag.config import (
        RAG_CANDIDATE_K,
        RAG_COHERE_API_KEY,
        RAG_COHERE_RERANK_MODEL,
        RAG_QUERY_TOP_K,
        RAG_RERANK_TOP_N,
    )
    from src.rag.retrieval.retrieve import collect_rerank_candidates
# candidates are nothing but top_k retrieved chinks from both the searches

def rerank_with_cohere(
    query: str,
    candidates: list[dict[str, Any]],
    api_key: str,
    model: str = "rerank-v3.5",
    top_n: int | None = None,
    max_tokens_per_doc: int = 4096,
    timeout: int = 30,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rerank retrieved candidates using Cohere's v2 rerank endpoint."""
    if not candidates:
        return [], {"status": "skipped", "detail": "No candidates to rerank."}

    if not api_key.strip():
        return candidates, {"status": "skipped", "detail": "RAG_COHERE_API_KEY is not set."}

    client = ClientV2(api_key=api_key.strip(), timeout=timeout)
    response = client.rerank(
        model=model,
        query=query,
        documents=[str(candidate.get("document", "")) for candidate in candidates],
        top_n=top_n or len(candidates),
        max_tokens_per_doc=max_tokens_per_doc,
    )

    reranked: list[dict[str, Any]] = []
    for rank, result in enumerate(response.results, start=1):
        candidate_index = result.index
        candidate = dict(candidates[candidate_index])
        candidate["rerank_score"] = float(result.relevance_score)
        candidate["rerank_rank"] = rank
        candidate["rerank_model"] = model
        reranked.append(candidate)

    return reranked, {"status": "applied", "detail": f"model={model}"}


def retrieve_and_rerank(
    query: str,
    top_k: int = RAG_QUERY_TOP_K,
    candidate_k: int = RAG_CANDIDATE_K,
    api_key: str = RAG_COHERE_API_KEY,
    model: str = RAG_COHERE_RERANK_MODEL,
    rerank_top_n: int = RAG_RERANK_TOP_N,
    max_tokens_per_doc: int = 4096,
    timeout: int = 30,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collect retrieval candidates and return the final Cohere-ranked chunks."""
    candidates = collect_rerank_candidates(
        query=query,
        top_k=candidate_k,
        candidate_k=candidate_k,
    )
    if not candidates:
        return [], {"status": "skipped", "detail": "No candidates retrieved for reranking."}

    reranked_results, status = rerank_with_cohere(
        query=query,
        candidates=candidates,
        api_key=api_key,
        model=model,
        top_n=min(rerank_top_n, len(candidates)),
        max_tokens_per_doc=max_tokens_per_doc,
        timeout=timeout,
    )
    return reranked_results[:top_k], status
