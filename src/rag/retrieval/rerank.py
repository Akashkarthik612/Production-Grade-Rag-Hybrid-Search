"""Optional Cohere reranking helpers for second-stage ranking."""

from __future__ import annotations

from typing import Any

import requests


COHERE_RERANK_URL = "https://api.cohere.com/v2/rerank"


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

    payload: dict[str, Any] = {
        "model": model,
        "query": query,
        "documents": [str(candidate.get("document", "")) for candidate in candidates],
        "top_n": top_n or len(candidates),
        "max_tokens_per_doc": max_tokens_per_doc,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    response = requests.post(
        COHERE_RERANK_URL,
        json=payload,
        headers=headers,
        timeout=timeout,
    )
    response.raise_for_status()
    body = response.json()

    reranked: list[dict[str, Any]] = []
    for rank, result in enumerate(body.get("results", []), start=1):
        candidate_index = result["index"]
        candidate = dict(candidates[candidate_index])
        candidate["rerank_score"] = float(result["relevance_score"])
        candidate["rerank_rank"] = rank
        candidate["rerank_model"] = model
        reranked.append(candidate)

    return reranked, {"status": "applied", "detail": f"model={model}"}
