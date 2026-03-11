from pathlib import Path
import sys

try:
    from src.rag.config import (
        RAG_CHROMA_PATH,
        RAG_COLLECTION_NAME,
        RAG_EMBEDDING_PROVIDER,
        RAG_INGEST_VERSION,
    )
    from src.rag.ingestion.embed_store import _embed_texts, _get_chroma_client
except ModuleNotFoundError:
    # Allow running this file directly: `python src/scripts/query_check.py`.
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.rag.config import (
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


def run_query_check(queries: list[str], top_k: int = 3) -> None:
    client, storage_mode = _get_chroma_client(chroma_path=RAG_CHROMA_PATH)
    collection = client.get_or_create_collection(name=RAG_COLLECTION_NAME)
    total = collection.count()
    version_rows = collection.get(where={"ingest_version": RAG_INGEST_VERSION}, include=[])
    version_total = len(version_rows.get("ids", []))

    print(
        f"Collection='{RAG_COLLECTION_NAME}' storage='{storage_mode}' "
        f"embedding_provider='{RAG_EMBEDDING_PROVIDER}' docs={total}"
    )
    print(f"Chunks for ingest_version='{RAG_INGEST_VERSION}': {version_total}")

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
        query_embeddings, provider_used = _embed_texts([query], provider=RAG_EMBEDDING_PROVIDER)
        result = collection.query(
            query_embeddings=query_embeddings,
            n_results=top_k,
            where={"ingest_version": RAG_INGEST_VERSION},
            include=["documents", "metadatas", "distances"],
        )

        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        print("\n" + "=" * 90)
        print(f"QUERY: {query}")
        print(
            f"provider_used={provider_used} | top_k={top_k} "
            f"| ingest_version={RAG_INGEST_VERSION}"
        )

        if not docs:
            print("No results returned.")
            continue

        for idx, (doc, meta, dist) in enumerate(zip(docs, metas, distances), start=1):
            paper_id = (meta or {}).get("paper_id", "unknown")
            chunk_index = (meta or {}).get("chunk_index", "unknown")
            source = (meta or {}).get("source", "unknown")
            snippet = str(doc).replace("\n", " ").strip()[:260]
            print(f"\n[{idx}] distance={dist:.6f} paper_id={paper_id} chunk={chunk_index}")
            print(f"source={source}")
            print(f"snippet={snippet}...")


def main() -> None:
    run_query_check(DEFAULT_QUERIES, top_k=3)


if __name__ == "__main__":
    main()
