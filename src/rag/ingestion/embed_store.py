"""Embed chunked documents and store vectors in ChromaDB."""

import hashlib
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from chromadb.errors import InternalError

try:
    from src.rag.ingestion.chunking import chunk_text
    from src.rag.ingestion.inout_ingestion import load_documents_from_local
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from src.rag.ingestion.chunking import chunk_text
    from src.rag.ingestion.inout_ingestion import load_documents_from_local

# vector creation below is a simple hash-based embedding for demonstration and fallback purposes.
def _hash_embed(text: str, dim: int = 256) -> list[float]:
    vector = [0.0] * dim
    for token in text.lower().split():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:2], "big") % dim
        sign = 1.0 if digest[2] % 2 == 0 else -1.0
        vector[index] += sign
    return vector

# will embed the text based on hash or chroma's default embedding function, depending on the provider argument.
def _embed_texts(texts: list[str], provider: str = "default") -> tuple[list[list[float]], str]:
    if provider == "hash":
        return [_hash_embed(text) for text in texts], "hash"

    if provider == "default":
        try:
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

            ef = DefaultEmbeddingFunction()
            return ef(texts), "default"
        except Exception:
            return [_hash_embed(text) for text in texts], "hash"

    raise ValueError("provider must be 'default' or 'hash'")


def _get_chroma_client(chroma_path: str = ".chroma") -> tuple[chromadb.ClientAPI, str]:
    db_path = Path(chroma_path)
    db_path.mkdir(parents=True, exist_ok=True)

    try:
        return (
            chromadb.PersistentClient(
                path=str(db_path), settings=Settings(anonymized_telemetry=False)
            ),
            "persistent",
        )
    except InternalError:
        return chromadb.Client(settings=Settings(anonymized_telemetry=False)), "in-memory"


def _build_chunk_records(
    documents: list[dict[str, Any]],
    min_tokens: int = 500,
    max_tokens: int = 800,
    overlap: int = 80,
    ingest_version: str = "v1",
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for doc in documents:
        doc_id = str(doc.get("id", "")).strip()
        text = str(doc.get("text", "")).strip()
        if not text:
            continue

        chunks = chunk_text(
            text,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            overlap=overlap,
        )

        for chunk_index, ctext in enumerate(chunks):
            ctext = ctext.strip()
            if not ctext:
                continue

            stable_id = hashlib.sha1(
                f"{doc_id}|{chunk_index}|{ctext}".encode("utf-8")
            ).hexdigest()

            metadata = {
                "paper_id": doc_id,
                "chunk_index": chunk_index,
                "source": str(doc.get("source", "")),
                "ingest_version": ingest_version,
            }

            extra_metadata = doc.get("metadata", {})
            if isinstance(extra_metadata, dict):
                for key, value in extra_metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        metadata[key] = value

            records.append({"id": stable_id, "text": ctext, "metadata": metadata})

    return records


def _batch(items: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def index_documents_in_chroma(
    documents: list[dict[str, Any]],
    collection_name: str = "research_papers",
    chroma_path: str = ".chroma",
    embedding_provider: str = "default",
    min_tokens: int = 500,
    max_tokens: int = 800,
    overlap: int = 80,
    batch_size: int = 32,
    ingest_version: str = "v1",
    reset_collection: bool = False,
) -> dict[str, Any]:
    client, storage_mode = _get_chroma_client(chroma_path=chroma_path)
    if reset_collection:
        try:
            client.delete_collection(name=collection_name)
        except Exception:
            pass
    collection = client.get_or_create_collection(name=collection_name)

    records = _build_chunk_records(
        documents,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        overlap=overlap,
        ingest_version=ingest_version,
    )

    if not records:
        return {
            "documents": len(documents),
            "chunks_indexed": 0,
            "collection": collection_name,
            "storage_mode": storage_mode,
            "embedding_provider": embedding_provider,
        }

    unique_ids = {record["id"] for record in records}
    if len(unique_ids) != len(records):
        raise ValueError("Duplicate chunk IDs found. Check input documents.")

    provider_used = embedding_provider
    for batch in _batch(records, batch_size=batch_size):
        ids = [item["id"] for item in batch]
        docs = [item["text"] for item in batch]
        metadatas = [item["metadata"] for item in batch]

        embeddings, provider_used = _embed_texts(docs, provider=provider_used)
        collection.upsert(ids=ids, documents=docs, metadatas=metadatas, embeddings=embeddings)

    return {
        "documents": len(documents),
        "chunks_indexed": len(records),
        "collection": collection_name,
        "collection_count": collection.count(),
        "storage_mode": storage_mode,
        "embedding_provider": provider_used,
    }


def index_local_papers(
    papers_dir: str,
    collection_name: str = "research_papers",
    chroma_path: str = ".chroma",
    embedding_provider: str = "default",
    min_tokens: int = 500,
    max_tokens: int = 800,
    overlap: int = 80,
    batch_size: int = 32,
    ingest_version: str = "v1",
    reset_collection: bool = False,
) -> dict[str, Any]:
    documents = load_documents_from_local(papers_dir)
    return index_documents_in_chroma(
        documents,
        collection_name=collection_name,
        chroma_path=chroma_path,
        embedding_provider=embedding_provider,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        overlap=overlap,
        batch_size=batch_size,
        ingest_version=ingest_version,
        reset_collection=reset_collection,
    )


if __name__ == "__main__":
    summary = index_local_papers(papers_dir="papers")
    print(summary)
