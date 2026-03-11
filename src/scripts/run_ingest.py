import math
import os
from pathlib import Path
import sys

try:
    import tiktoken
    from src.rag.ingestion.chunking import chunk_text
    from src.rag.ingestion.embed_store import (
        _batch,
        _build_chunk_records,
        _embed_texts,
        _get_chroma_client,
        index_local_papers,
    )
    from src.rag.ingestion.inout_ingestion import load_documents_from_local
    from src.rag.config import (
        RAG_BATCH_SIZE,
        RAG_CHROMA_PATH,
        RAG_COLLECTION_NAME,
        RAG_EMBEDDING_PROVIDER,
        RAG_INGEST_VERSION,
        RAG_MAX_TOKENS,
        RAG_MIN_TOKENS,
        RAG_OVERLAP,
        RAG_PAPERS_DIR,
    )
except ModuleNotFoundError:
    # Allow running this file directly: `python src/scripts/run_ingest.py`.
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    import tiktoken
    from src.rag.ingestion.chunking import chunk_text
    from src.rag.ingestion.embed_store import (
        _batch,
        _build_chunk_records,
        _embed_texts,
        _get_chroma_client,
        index_local_papers,
    )
    from src.rag.ingestion.inout_ingestion import load_documents_from_local
    from src.rag.config import (
        RAG_BATCH_SIZE,
        RAG_CHROMA_PATH,
        RAG_COLLECTION_NAME,
        RAG_EMBEDDING_PROVIDER,
        RAG_INGEST_VERSION,
        RAG_MAX_TOKENS,
        RAG_MIN_TOKENS,
        RAG_OVERLAP,
        RAG_PAPERS_DIR,
    )


def _as_bool(env_value: str | None, default: bool = False) -> bool:
    if env_value is None:
        return default
    return env_value.strip().lower() in {"1", "true", "yes", "y", "on"}


def run_verbose_pipeline(reset_collection: bool = False) -> dict:
    enc = tiktoken.get_encoding("cl100k_base")
    documents = load_documents_from_local(RAG_PAPERS_DIR)
    print(f"Loaded {len(documents)} documents from '{RAG_PAPERS_DIR}'")

    for doc_index, doc in enumerate(documents, start=1):
        doc_text = str(doc.get("text", ""))
        chunks = chunk_text(
            doc_text,
            min_tokens=RAG_MIN_TOKENS,
            max_tokens=RAG_MAX_TOKENS,
            overlap=RAG_OVERLAP,
        )
        token_counts = [len(enc.encode(chunk)) for chunk in chunks]
        total_tokens = len(enc.encode(doc_text))
        min_chunk_tokens = min(token_counts) if token_counts else 0
        max_chunk_tokens = max(token_counts) if token_counts else 0
        print(
            f"[DOC {doc_index}] source={doc.get('source')} total_tokens={total_tokens} "
            f"chunks={len(chunks)} min_chunk_tokens={min_chunk_tokens} max_chunk_tokens={max_chunk_tokens}"
        )

    records = _build_chunk_records(
        documents,
        min_tokens=RAG_MIN_TOKENS,
        max_tokens=RAG_MAX_TOKENS,
        overlap=RAG_OVERLAP,
        ingest_version=RAG_INGEST_VERSION,
    )
    print(f"Prepared {len(records)} chunk records")

    client, storage_mode = _get_chroma_client(chroma_path=RAG_CHROMA_PATH)
    if reset_collection:
        try:
            client.delete_collection(name=RAG_COLLECTION_NAME)
            print(f"Reset existing collection: {RAG_COLLECTION_NAME}")
        except Exception:
            pass
    collection = client.get_or_create_collection(name=RAG_COLLECTION_NAME)

    unique_ids = {record["id"] for record in records}
    if len(unique_ids) != len(records):
        raise ValueError("Duplicate chunk IDs found. Check input documents.")

    provider_used = RAG_EMBEDDING_PROVIDER
    batches = _batch(records, batch_size=RAG_BATCH_SIZE)
    total_batches = max(1, math.ceil(len(records) / RAG_BATCH_SIZE))
    print(
        f"Embedding provider={provider_used} | storage={storage_mode} | "
        f"collection={RAG_COLLECTION_NAME} | batches={total_batches}"
    )

    for batch_idx, batch in enumerate(batches, start=1):
        ids = [item["id"] for item in batch]
        # Explicitly keep chunked texts in `chunking_docs` before embedding.
        chunking_docs = [item["text"] for item in batch]
        metadatas = [item["metadata"] for item in batch]

        embeddings, provider_used = _embed_texts(chunking_docs, provider=provider_used)
        emb_dim = len(embeddings[0]) if embeddings else 0
        token_counts = [len(enc.encode(text)) for text in chunking_docs]
        avg_tokens = (sum(token_counts) / len(token_counts)) if token_counts else 0.0
        print(
            f"[BATCH {batch_idx}/{total_batches}] size={len(batch)} emb_dim={emb_dim} "
            f"avg_chunk_tokens={avg_tokens:.1f}"
        )

        collection.upsert(
            ids=ids,
            documents=chunking_docs,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    summary = {
        "documents": len(documents),
        "chunks_indexed": len(records),
        "collection": RAG_COLLECTION_NAME,
        "collection_count": collection.count(),
        "storage_mode": storage_mode,
        "embedding_provider": provider_used,
    }
    return summary


def main() -> None:
    verbose_mode = _as_bool(os.getenv("RAG_DEBUG_INGEST"), default=False)
    reset_collection = _as_bool(os.getenv("RAG_RESET_COLLECTION"), default=False)
    print(
        "Ingest config: "
        f"cwd='{Path.cwd()}' papers_dir='{RAG_PAPERS_DIR}' "
        f"collection='{RAG_COLLECTION_NAME}' chroma_path='{RAG_CHROMA_PATH}' "
        f"ingest_version='{RAG_INGEST_VERSION}'"
    )
    if verbose_mode:
        summary = run_verbose_pipeline(reset_collection=reset_collection)
    else:
        summary = index_local_papers(
            papers_dir=RAG_PAPERS_DIR,
            collection_name=RAG_COLLECTION_NAME,
            chroma_path=RAG_CHROMA_PATH,
            embedding_provider=RAG_EMBEDDING_PROVIDER,
            min_tokens=RAG_MIN_TOKENS,
            max_tokens=RAG_MAX_TOKENS,
            overlap=RAG_OVERLAP,
            batch_size=RAG_BATCH_SIZE,
            ingest_version=RAG_INGEST_VERSION,
            reset_collection=reset_collection,
        )
    print(f"Ingestion complete: {summary}")


if __name__ == "__main__":
    main()
