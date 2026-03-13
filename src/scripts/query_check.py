from pathlib import Path
import sys
from typing import Any

try:
    from src.rag.config import (
        RAG_CANDIDATE_K,
        RAG_CHROMA_PATH,
        RAG_COHERE_API_KEY,
        RAG_COHERE_RERANK_MODEL,
        RAG_COLLECTION_NAME,
        RAG_EMBEDDING_PROVIDER,
        RAG_INGEST_VERSION,
        RAG_QUERY_TOP_K,
        RAG_RERANK_ENABLED,
        RAG_RERANK_TOP_N,
        RAG_RRF_K,
    )
    from src.rag.generation.cite_answer import build_query_report, write_query_results_docx
    from src.rag.ingestion.embed_store import _get_chroma_client
    from src.rag.retrieval.rerank import rerank_with_cohere
    from src.rag.retrieval.retrieve import hybrid_search
except ModuleNotFoundError:
    # Allow running this file directly: `python src/scripts/query_check.py`.
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.rag.config import (
        RAG_CANDIDATE_K,
        RAG_CHROMA_PATH,
        RAG_COHERE_API_KEY,
        RAG_COHERE_RERANK_MODEL,
        RAG_COLLECTION_NAME,
        RAG_EMBEDDING_PROVIDER,
        RAG_INGEST_VERSION,
        RAG_QUERY_TOP_K,
        RAG_RERANK_ENABLED,
        RAG_RERANK_TOP_N,
        RAG_RRF_K,
    )
    from src.rag.generation.cite_answer import build_query_report, write_query_results_docx
    from src.rag.ingestion.embed_store import _get_chroma_client
    from src.rag.retrieval.rerank import rerank_with_cohere
    from src.rag.retrieval.retrieve import hybrid_search


DEFAULT_QUERIES = [
    "What is the transformer architecture?",
    "How is self-attention computed?",
    "What are the main contributions of this paper?",
]


def run_query_check(
    queries: list[str],
    top_k: int = RAG_QUERY_TOP_K,
    candidate_k: int = RAG_CANDIDATE_K,
    output_path: str = "artifacts/hybrid_answer_report.docx",
) -> None:
    client, storage_mode = _get_chroma_client(chroma_path=RAG_CHROMA_PATH)
    collection = client.get_or_create_collection(name=RAG_COLLECTION_NAME)
    total = collection.count()
    version_rows = collection.get(where={"ingest_version": RAG_INGEST_VERSION}, include=[])
    version_total = len(version_rows.get("ids", []))
    reports: list[dict[str, Any]] = []

    print(
        f"Collection='{RAG_COLLECTION_NAME}' storage='{storage_mode}' "
        f"embedding_provider='{RAG_EMBEDDING_PROVIDER}' docs={total}"
    )
    print(f"Chunks for ingest_version='{RAG_INGEST_VERSION}': {version_total}")
    if storage_mode == "in-memory":
        print(
            "Warning: Chroma fell back to in-memory mode. "
            "Your persisted index was not opened, so retrieval will not see stored chunks."
        )

    if total == 0:
        print("No chunks found. Run ingestion first: python -m src.scripts.run_ingest")
        return
    if version_total == 0:
        print(
            "No chunks match current RAG_INGEST_VERSION. "
            "Set the same RAG_INGEST_VERSION for ingest and query, or re-ingest."
        )
        return

    for query in queries:
        hybrid_results = hybrid_search(
            query=query,
            top_k=candidate_k,
            candidate_k=candidate_k,
            rrf_k=RAG_RRF_K,
        )

        rerank_status = {
            "status": "skipped",
            "detail": "Reranker disabled. Set RAG_RERANK_ENABLED=1 to enable Cohere reranking.",
        }
        final_results = hybrid_results[:top_k]
        if RAG_RERANK_ENABLED:
            try:
                reranked_results, rerank_status = rerank_with_cohere(
                    query=query,
                    candidates=hybrid_results,
                    api_key=RAG_COHERE_API_KEY,
                    model=RAG_COHERE_RERANK_MODEL,
                    top_n=min(RAG_RERANK_TOP_N, len(hybrid_results)),
                )
                final_results = reranked_results[:top_k]
            except Exception as exc:
                rerank_status = {"status": "error", "detail": str(exc)}

        print("\n" + "=" * 90)
        print(f"QUERY: {query}")
        print(
            f"candidate_k={candidate_k} | top_k={top_k} | ingest_version={RAG_INGEST_VERSION}"
        )

        if not final_results:
            print("No results returned.")
            reports.append(build_query_report(query=query, results=[], rerank_status=rerank_status))
            continue

        report = build_query_report(
            query=query,
            results=final_results,
            rerank_status=rerank_status,
        )
        reports.append(report)

        print("ANSWER:")
        print(report["answer"])
        print("\nCITATIONS:")
        for citation in report["citations"]:
            print(citation)
        print("\nPIPELINE:")
        print(f"hybrid_candidates={len(hybrid_results)} | reranker={rerank_status['status']}")
        if rerank_status.get("detail"):
            print(rerank_status["detail"])

    saved_path = write_query_results_docx(
        output_path=output_path,
        reports=reports,
        title="Hybrid Answer Report",
    )
    print("\n" + "=" * 90)
    print(f"Saved Word report to: {saved_path}")


def main() -> None:
    run_query_check(DEFAULT_QUERIES, top_k=RAG_QUERY_TOP_K, candidate_k=RAG_CANDIDATE_K)


if __name__ == "__main__":
    main()
