"""Embed chunked documents and store vectors in ChromaDB."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

try:
    from src.rag.ingestion.chunking import chunk_text
    from src.rag.ingestion.inout_ingestion import load_documents_from_local
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from src.rag.ingestion.chunking import chunk_text
    from src.rag.ingestion.inout_ingestion import load_documents_from_local


_CHROMA_CLIENT_CACHE: dict[tuple[str, str], chromadb.ClientAPI] = {}
_INDEX_STATE_FILENAME = ".rag_index_state.json"


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
    resolved_path = str(db_path.resolve())

    try:
        cache_key = ("persistent", resolved_path)
        if cache_key not in _CHROMA_CLIENT_CACHE:
            _CHROMA_CLIENT_CACHE[cache_key] = chromadb.PersistentClient(
                path=str(db_path), settings=Settings(anonymized_telemetry=False)
            )
        return _CHROMA_CLIENT_CACHE[cache_key], "persistent"
    except Exception:
        cache_key = ("in-memory", resolved_path)
        if cache_key not in _CHROMA_CLIENT_CACHE:
            _CHROMA_CLIENT_CACHE[cache_key] = chromadb.Client(
                settings=Settings(anonymized_telemetry=False)
            )
        return _CHROMA_CLIENT_CACHE[cache_key], "in-memory"


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


def _hash_file_contents(path: str) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _build_source_manifest(
    papers_dir: str,
    allowed_extensions: tuple[str, ...] = (".txt", ".md", ".pdf"),
) -> dict[str, Any]:
    base_dir = Path(papers_dir)
    if not base_dir.is_dir():
        raise FileNotFoundError(f"papers_dir not found: {papers_dir}")

    files: list[dict[str, Any]] = []
    for root, _dirs, file_names in os.walk(base_dir):
        for file_name in sorted(file_names):
            file_path = Path(root) / file_name
            ext = file_path.suffix.lower()
            if ext not in allowed_extensions:
                continue

            stat = file_path.stat()
            files.append(
                {
                    "relative_path": file_path.relative_to(base_dir).as_posix(),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "sha1": _hash_file_contents(str(file_path)),
                }
            )

    files.sort(key=lambda item: item["relative_path"])
    return {
        "papers_dir": str(base_dir.resolve()),
        "file_count": len(files),
        "files": files,
    }


def _index_state_path(chroma_path: str) -> Path:
    return Path(chroma_path) / _INDEX_STATE_FILENAME


def _load_index_state(chroma_path: str) -> dict[str, Any] | None:
    state_path = _index_state_path(chroma_path)
    if not state_path.exists():
        return None

    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_index_state(chroma_path: str, state: dict[str, Any]) -> None:
    state_path = _index_state_path(chroma_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _build_index_state(
    papers_dir: str,
    collection_name: str,
    chroma_path: str,
    embedding_provider: str,
    ingest_version: str,
) -> dict[str, Any]:
    return {
        "collection_name": collection_name,
        "chroma_path": str(Path(chroma_path).resolve()),
        "embedding_provider": embedding_provider,
        "ingest_version": ingest_version,
        "manifest": _build_source_manifest(papers_dir),
    }


def _has_indexed_chunks(collection_name: str, chroma_path: str, ingest_version: str) -> bool:
    client, _storage_mode = _get_chroma_client(chroma_path=chroma_path)
    collection = client.get_or_create_collection(name=collection_name)
    rows = collection.get(where={"ingest_version": ingest_version}, include=[])
    return bool(rows.get("ids", []))


def inspect_index_state(
    papers_dir: str,
    collection_name: str = "research_papers",
    chroma_path: str = ".chroma",
    embedding_provider: str = "default",
    ingest_version: str = "v1",
) -> dict[str, Any]:
    expected_state = _build_index_state(
        papers_dir=papers_dir,
        collection_name=collection_name,
        chroma_path=chroma_path,
        embedding_provider=embedding_provider,
        ingest_version=ingest_version,
    )
    saved_state = _load_index_state(chroma_path)

    if not _has_indexed_chunks(collection_name, chroma_path, ingest_version):
        return {
            "needs_reindex": True,
            "reason": "missing_index",
            "expected_state": expected_state,
        }

    if saved_state is None:
        return {
            "needs_reindex": True,
            "reason": "missing_state",
            "expected_state": expected_state,
        }

    if saved_state.get("ingest_version") != ingest_version:
        return {
            "needs_reindex": True,
            "reason": "ingest_version_changed",
            "expected_state": expected_state,
        }

    if saved_state.get("manifest") != expected_state.get("manifest"):
        return {
            "needs_reindex": True,
            "reason": "documents_changed",
            "expected_state": expected_state,
        }

    return {
        "needs_reindex": False,
        "reason": "up_to_date",
        "expected_state": expected_state,
        "saved_state": saved_state,
    }


def ensure_local_papers_index(
    papers_dir: str,
    collection_name: str = "research_papers",
    chroma_path: str = ".chroma",
    embedding_provider: str = "default",
    min_tokens: int = 500,
    max_tokens: int = 800,
    overlap: int = 80,
    batch_size: int = 32,
    ingest_version: str = "v1",
) -> dict[str, Any]:
    status = inspect_index_state(
        papers_dir=papers_dir,
        collection_name=collection_name,
        chroma_path=chroma_path,
        embedding_provider=embedding_provider,
        ingest_version=ingest_version,
    )
    if not status["needs_reindex"]:
        client, storage_mode = _get_chroma_client(chroma_path=chroma_path)
        collection = client.get_or_create_collection(name=collection_name)
        return {
            "action": "reused",
            "reason": status["reason"],
            "documents": status["saved_state"]["manifest"]["file_count"],
            "chunks_indexed": None,
            "collection": collection_name,
            "collection_count": collection.count(),
            "storage_mode": storage_mode,
            "embedding_provider": embedding_provider,
        }

    summary = index_local_papers(
        papers_dir=papers_dir,
        collection_name=collection_name,
        chroma_path=chroma_path,
        embedding_provider=embedding_provider,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        overlap=overlap,
        batch_size=batch_size,
        ingest_version=ingest_version,
        reset_collection=True,
    )
    _write_index_state(chroma_path, status["expected_state"])
    summary["action"] = "indexed"
    summary["reason"] = status["reason"]
    return summary


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
