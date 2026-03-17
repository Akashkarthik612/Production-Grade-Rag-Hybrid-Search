import argparse
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
        RAG_PAPERS_DIR,
        RAG_QUERY_TOP_K,
        RAG_RERANK_ENABLED,
        RAG_RERANK_TOP_N,
    )
    from src.rag.generation.cite_answer import build_query_report, write_query_results_docx
    from src.rag.ingestion.embed_store import _get_chroma_client, ensure_local_papers_index
    from src.rag.retrieval.rerank import retrieve_and_rerank
    from src.rag.retrieval.retrieve import (
        collect_rerank_candidates,
        invalidate_retrieval_cache,
        warm_retrieval_cache,
    )
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
        RAG_PAPERS_DIR,
        RAG_QUERY_TOP_K,
        RAG_RERANK_ENABLED,
        RAG_RERANK_TOP_N,
    )
    from src.rag.generation.cite_answer import build_query_report, write_query_results_docx
    from src.rag.ingestion.embed_store import _get_chroma_client, ensure_local_papers_index
    from src.rag.retrieval.rerank import retrieve_and_rerank
    from src.rag.retrieval.retrieve import (
        collect_rerank_candidates,
        invalidate_retrieval_cache,
        warm_retrieval_cache,
    )


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
    startup_summary = ensure_local_papers_index(
        papers_dir=RAG_PAPERS_DIR,
        collection_name=RAG_COLLECTION_NAME,
        chroma_path=RAG_CHROMA_PATH,
        embedding_provider=RAG_EMBEDDING_PROVIDER,
        ingest_version=RAG_INGEST_VERSION,
    )
    if startup_summary["action"] == "indexed":
        invalidate_retrieval_cache(
            chroma_path=RAG_CHROMA_PATH,
            collection_name=RAG_COLLECTION_NAME,
            ingest_version=RAG_INGEST_VERSION,
        )
    cache_summary = warm_retrieval_cache(
        chroma_path=RAG_CHROMA_PATH,
        collection_name=RAG_COLLECTION_NAME,
        ingest_version=RAG_INGEST_VERSION,
    )

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
    print(
        f"Startup index action='{startup_summary['action']}' reason='{startup_summary['reason']}' "
        f"| bm25_cache='{cache_summary['status']}' chunks={cache_summary['chunk_count']}"
    )
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
        candidate_results = collect_rerank_candidates(
            query=query,
            top_k=candidate_k,
            candidate_k=candidate_k,
        )

        rerank_status = {
            "status": "skipped",
            "detail": "Reranker disabled. Set RAG_RERANK_ENABLED=1 to enable Cohere reranking.",
        }
        final_results = candidate_results[:top_k]
        if RAG_RERANK_ENABLED:
            try:
                final_results, rerank_status = retrieve_and_rerank(
                    query=query,
                    top_k=top_k,
                    candidate_k=candidate_k,
                    api_key=RAG_COHERE_API_KEY,
                    model=RAG_COHERE_RERANK_MODEL,
                    rerank_top_n=RAG_RERANK_TOP_N,
                )
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
        print(f"rerank_candidates={len(candidate_results)} | reranker={rerank_status['status']}")
        if rerank_status.get("detail"):
            print(rerank_status["detail"])

    saved_path = write_query_results_docx(
        output_path=output_path,
        reports=reports,
        title="Hybrid Answer Report",
    )
    print("\n" + "=" * 90)
    print(f"Saved Word report to: {saved_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAG query checks against the indexed collection.")
    parser.add_argument(
        "--query",
        dest="queries",
        action="append",
        help="Custom query to run. Repeat --query to run multiple queries.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=RAG_QUERY_TOP_K,
        help="Number of final results to keep.",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=RAG_CANDIDATE_K,
        help="Number of retrieval candidates to gather before reranking.",
    )
    parser.add_argument(
        "--output-path",
        default="artifacts/hybrid_answer_report.docx",
        help="Where to write the report document.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    queries = args.queries or DEFAULT_QUERIES
    run_query_check(
        queries,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
