# Production Grade RAG System

A small retrieval-augmented generation (RAG) project for experimenting with document ingestion, chunking, vector storage, and query-time retrieval.

This repository currently focuses on:

- ingesting local papers into ChromaDB
- chunking documents for retrieval
- embedding chunks and storing them in a vector collection
- running simple query checks against the indexed collection
- preparing CI-based evaluation for a future retrieval benchmark

## Project Structure

- `src/rag/`: core RAG configuration and ingestion logic
- `src/scripts/run_ingest.py`: ingest local documents into ChromaDB
- `src/scripts/query_check.py`: run sample queries and inspect retrieved chunks
- `tests/ci/`: placeholder location for CI-based query and answer evaluation

## Requirements

- Python 3.11 recommended
- dependencies from `requirements.txt`

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install pytest
```

## Configuration

Runtime configuration is defined in `src/rag/config.py` and can be overridden with environment variables.

Examples:

- `RAG_PAPERS_DIR`
- `RAG_COLLECTION_NAME`
- `RAG_CHROMA_PATH`
- `RAG_EMBEDDING_PROVIDER`
- `RAG_MIN_TOKENS`
- `RAG_MAX_TOKENS`
- `RAG_OVERLAP`
- `RAG_BATCH_SIZE`
- `RAG_INGEST_VERSION`
- `RAG_RERANK_ENABLED`
- `RAG_COHERE_API_KEY`
- `RAG_COHERE_RERANK_MODEL`

Copy `.env.example` to `.env` and adjust values for your machine. Cohere reranking is optional and stays off by default unless both `RAG_RERANK_ENABLED=1` and a non-empty `RAG_COHERE_API_KEY` are set.

## Run Ingestion

Index local papers into Chroma:

```bash
python -m src.scripts.run_ingest
```

For direct script execution:

```bash
python src/scripts/run_ingest.py
```

## Run Query Check

Inspect retrieved chunks for a few sample queries:

```bash
python -m src.scripts.query_check
```

Or:

```bash
python src/scripts/query_check.py
```

The query check currently prints:

- the query text
- the embedding provider used
- `top_k` retrieval count
- returned chunk snippets
- retrieval distances for each result

If Cohere reranking is enabled, the query check will also report rerank status for the final results.

## Cohere Keys

For demo use, a Cohere trial or evaluation key is enough to test reranking. Keep that key in your local environment or `.env` and never commit it to the repository.

For production, use a paid Cohere key or your clients' own keys. Trial keys are suitable for low-volume demos and development, but they have lower limits and are not intended for sustained production traffic.

## CI

GitHub Actions CI is configured in `.github/workflows/ci.yml`.

The current workflow:

- installs project dependencies
- installs `pytest`
- runs the test suite
- allows an empty placeholder test file while evaluation code is still being added

## Next Steps

Planned improvements include:

- query-answer evaluation with 20 to 30 manually created test cases
- retrieval metrics such as `Hit@k` and `Recall@k`
- hybrid search with BM25 plus vector retrieval
- answer quality benchmarking

## Status

This is an active work-in-progress repository. The README is intentionally minimal and can be expanded later with architecture diagrams, benchmarks, and usage examples.
