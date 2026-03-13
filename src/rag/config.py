"""Runtime settings for ingestion pipeline."""

import os


def _as_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


RAG_PAPERS_DIR = os.getenv("RAG_PAPERS_DIR", "papers")
RAG_COLLECTION_NAME = os.getenv("RAG_COLLECTION_NAME", "research_papers")
RAG_CHROMA_PATH = os.getenv("RAG_CHROMA_PATH", ".chroma")
RAG_EMBEDDING_PROVIDER = os.getenv("RAG_EMBEDDING_PROVIDER", "default")   # which ever embedding we need we can choose this
RAG_MIN_TOKENS = int(os.getenv("RAG_MIN_TOKENS", "500"))
RAG_MAX_TOKENS = int(os.getenv("RAG_MAX_TOKENS", "800"))
RAG_OVERLAP = int(os.getenv("RAG_OVERLAP", "80"))
RAG_BATCH_SIZE = int(os.getenv("RAG_BATCH_SIZE", "32"))
RAG_INGEST_VERSION = os.getenv("RAG_INGEST_VERSION", "v1")
RAG_QUERY_TOP_K = int(os.getenv("RAG_QUERY_TOP_K", "3"))
RAG_CANDIDATE_K = int(os.getenv("RAG_CANDIDATE_K", "8"))
RAG_RRF_K = int(os.getenv("RAG_RRF_K", "60"))
RAG_RERANK_ENABLED = _as_bool("RAG_RERANK_ENABLED", "0")
RAG_RERANK_TOP_N = int(os.getenv("RAG_RERANK_TOP_N", "3"))
RAG_COHERE_API_KEY = os.getenv("RAG_COHERE_API_KEY", "")
RAG_COHERE_RERANK_MODEL = os.getenv("RAG_COHERE_RERANK_MODEL", "rerank-v3.5")
