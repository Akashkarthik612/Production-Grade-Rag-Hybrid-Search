"""CI query evaluation for the retrieval and rerank pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path

from src.rag.generation.cite_answer import build_query_report
from src.rag.ingestion.embed_store import index_local_papers
from src.rag.retrieval import retrieve as retrieve_module
from src.rag.retrieval.rerank import retrieve_and_rerank


DEFAULT_QUERIES = [
    "What core architectural components do Transformers use to model sequences instead of recurrent or convolutional neural networks?",
    "In the standard Transformer encoder, what are the two main sub-layers found in each encoding block?",
    "Why is parallelization difficult in recurrent models compared to Transformers?",
    "What specific mechanism does the Transformer use to distinguish the order of words in a sequence?",
    "What BLEU score did the original Transformer achieve on the WMT 2014 English-to-German translation task?",
    "How does the attention mechanism in Transformers shorten the computational distance between two inputs compared to RNNs?",
    "What are the three components of the QKV attention model?",
    "In the post-norm Transformer, where is the residual connection created relative to layer normalization?",
    "What is the purpose of multi-head attention in the Transformer?",
    "Which optimization method is commonly used for updating weights during Transformer training?",
]


def _prepare_ci_index() -> tuple[Path, str, str]:
    chroma_path = Path("artifacts/ci_eval_chroma")
    collection_name = "ci_eval_papers"
    ingest_version = "ci-eval"

    summary = index_local_papers(
        papers_dir="papers",
        collection_name=collection_name,
        chroma_path=str(chroma_path),
        embedding_provider="default",
        ingest_version=ingest_version,
        reset_collection=True,
    )
    print(f"INDEX SUMMARY: {summary}")
    return chroma_path, collection_name, ingest_version


def test_query_evaluation_pipeline(monkeypatch) -> None:
    chroma_path, collection_name, ingest_version = _prepare_ci_index()

    monkeypatch.setattr(retrieve_module, "RAG_CHROMA_PATH", str(chroma_path))
    monkeypatch.setattr(retrieve_module, "RAG_COLLECTION_NAME", collection_name)
    monkeypatch.setattr(retrieve_module, "RAG_INGEST_VERSION", ingest_version)
    monkeypatch.setattr(retrieve_module, "RAG_EMBEDDING_PROVIDER", "default")

    api_key = os.getenv("RAG_COHERE_API_KEY", "").strip()
    answers: list[dict[str, object]] = []

    for query in DEFAULT_QUERIES:
        results, rerank_status = retrieve_and_rerank(
            query=query,
            top_k=3,
            candidate_k=8,
            api_key=api_key,
        )
        report = build_query_report(query=query, results=results, rerank_status=rerank_status)
        answers.append(
            {
                "query": query,
                "answer": report["answer"],
                "citations": report["citations"],
                "rerank_status": rerank_status,
                "result_count": len(results),
                "top_result_metadata": results[0].get("metadata", {}) if results else {},
            }
        )

        print(f"QUERY: {query}")
        print(f"ANSWER: {report['answer']}")
        print(f"CITATIONS: {report['citations']}")
        print(f"RERANKER: {rerank_status['status']}")
        print("-" * 80)

        assert results, f"No results returned for query: {query}"
        assert report["answer"], f"Empty answer returned for query: {query}"
        assert results[0].get("document"), f"Top result document was empty for query: {query}"

    artifact_path = Path("artifacts/query_eval_answers.json")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(answers, indent=2), encoding="utf-8")

    assert len(answers) == len(DEFAULT_QUERIES)
