"""Runtime settings for ingestion pipeline."""

import os

RAG_PAPERS_DIR = os.getenv("RAG_PAPERS_DIR", "papers")
RAG_COLLECTION_NAME = os.getenv("RAG_COLLECTION_NAME", "research_papers")
RAG_CHROMA_PATH = os.getenv("RAG_CHROMA_PATH", ".chroma")
RAG_EMBEDDING_PROVIDER = os.getenv("RAG_EMBEDDING_PROVIDER", "default")   # which ever embedding we need we can choose this
RAG_MIN_TOKENS = int(os.getenv("RAG_MIN_TOKENS", "500"))
RAG_MAX_TOKENS = int(os.getenv("RAG_MAX_TOKENS", "800"))
RAG_OVERLAP = int(os.getenv("RAG_OVERLAP", "80"))
RAG_BATCH_SIZE = int(os.getenv("RAG_BATCH_SIZE", "32"))
RAG_INGEST_VERSION = os.getenv("RAG_INGEST_VERSION", "v1")
