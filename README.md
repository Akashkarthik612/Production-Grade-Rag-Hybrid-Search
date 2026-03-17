# Production Grade RAG System

A retrieval-augmented generation tool for enterprise documents such as research papers and financial documents.

This tool ingests PDFs, chunks them with token-aware logic, stores embeddings in ChromaDB, retrieves context with both lexical and vector search, optionally reranks candidates with Cohere, and produces citation-ready answers plus report artifacts.

## Why This Project Is Interesting

Most RAG demos stop at "embed documents and ask a question." This one goes further:

- token-aware ingestion for long technical papers
- persistent vector storage with ChromaDB
- hybrid retrieval using BM25 plus semantic search
- optional second-stage reranking with Cohere
- extractive answers with source citations
- CI-oriented evaluation tests and saved artifacts
- Docker support for repeatable ingestion workflows

In short, this is a production-minded RAG tool that takes raw documents all the way to cited answers.

## What It Does

This tool consists of three AI pipelines:

### 1. Ingestion And Chunking Pipeline

The first AI pipeline reads PDFs from `papers/`, extracts the text, cleans it, and breaks large documents into token-aware overlapping chunks so the knowledge base remains retrieval-ready even for long enterprise files.

### 2. Embedding, Storage, And Retrieval Pipeline

The second AI pipeline converts those chunks into embeddings, stores them in ChromaDB, and retrieves relevant context through a hybrid search layer that combines:

- BM25 lexical retrieval
- vector similarity search

This gives clients both precision on exact terminology and semantic recall for concept-level questions.

### 3. Reranking And Answer Pipeline

The third AI pipeline takes the retrieved candidates, applies an optional reranker at the final stage, selects the strongest context, and produces citation-ready answers with traceable source chunks.

The query workflow can also export `.docx` reports into `artifacts/` for review, delivery, or internal validation.

## Architecture At A Glance

```text
PDF papers
   |
   v
Text extraction
   |
   v
Token-aware chunking
   |
   v
Embedding + ChromaDB indexing
   |
   v
Hybrid retrieval (BM25 + vector)
   |
   v
Optional Cohere reranking
   |
   v
Extractive answer + citations + report artifact
```

## Tech Stack

- Python
- ChromaDB
- `tiktoken`
- `pypdf`
- `rank-bm25`
- Cohere rerank API
- Pytest
- Docker and Docker Compose

## Repository Layout

```text
src/
  rag/
    config.py                 Runtime configuration from env
    ingestion/                Loading, chunking, embedding, indexing
    retrieval/                BM25, vector search, reranking
    generation/               Cited answers and report generation
  scripts/
    run_ingest.py             End-to-end ingestion entrypoint
    query_check.py            Query pipeline and report export
    ask.py                    Query-facing script scaffold

papers/                       Source PDFs
artifacts/                    Generated reports and evaluation outputs
tests/ci/                     Retrieval and answer evaluation tests
Dockerfile                    Containerized ingestion setup
docker-compose.yml            Local container workflow
```

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install pytest
```

### 2. Configure environment

Copy `.env.example` to `.env` and adjust values if needed.

Important settings:

- `RAG_PAPERS_DIR`
- `RAG_COLLECTION_NAME`
- `RAG_CHROMA_PATH`
- `RAG_EMBEDDING_PROVIDER`
- `RAG_MIN_TOKENS`
- `RAG_MAX_TOKENS`
- `RAG_OVERLAP`
- `RAG_BATCH_SIZE`
- `RAG_INGEST_VERSION`
- `RAG_QUERY_TOP_K`
- `RAG_CANDIDATE_K`
- `RAG_RERANK_ENABLED`
- `RAG_RERANK_TOP_N`
- `RAG_COHERE_API_KEY`
- `RAG_COHERE_RERANK_MODEL`

By default, reranking stays off unless:

```env
RAG_RERANK_ENABLED=1
RAG_COHERE_API_KEY=your_key_here
```

### 3. Ingest the paper collection

```bash
python -m src.scripts.run_ingest
```

You can also run:

```bash
python src/scripts/run_ingest.py
```

### 4. Run hybrid retrieval and generate answers

```bash
python -m src.scripts.query_check
```

Or pass custom queries:

```bash
python -m src.scripts.query_check --query "What is self-attention?" --top-k 3 --candidate-k 8
```

This prints answers and citations to the console and saves a Word report to:

```text
artifacts/hybrid_answer_report.docx
```

## Example Workflow

1. Drop PDFs into `papers/`
2. Run ingestion to build the Chroma index
3. Ask research questions against the indexed papers
4. Inspect citations and returned chunks
5. Export reports for evaluation or presentation

This workflow is designed for document intelligence use cases where clients want grounded answers, retrieval transparency, and deliverable-ready outputs.

## Testing And Evaluation

The tool includes CI-style tests for retrieval and answer generation behavior.

Key evaluation ideas already reflected in the tool:

- hybrid retrieval result merging
- citation formatting
- answer construction from retrieved chunks
- reranking fallback behavior when no API key is present
- end-to-end query evaluation against indexed papers

Run tests with:

```bash
pytest -q
```

The evaluation workflow can also write JSON and Word artifacts into `artifacts/` for manual inspection.

## Docker Support

Build and run the ingestion pipeline in Docker:

```bash
docker compose up --build
```

The container mounts:

- `./papers` to `/app/papers`
- `./.chroma` to `/app/.chroma`

So your indexed data persists locally across runs.

## Outputs

Depending on the workflow you run, the tool can produce:

- persisted Chroma collections in `.chroma/`
- answer reports in `artifacts/*.docx`
- evaluation summaries in `artifacts/*.json`

## Notes On Embeddings

The system supports:

- Chroma's default embedding function
- a hash-based fallback embedding mode for resilience and local experimentation

That makes the pipeline easier to run even when a full embedding backend is unavailable.

## Notes On Reranking

Cohere reranking is optional. If the API key is missing, the pipeline does not fail; it gracefully skips reranking and returns the hybrid retrieval candidates directly.

That fallback behavior is especially useful for local development and CI runs.

## Project Status

This tool is positioned as a client-ready RAG solution with configurable ingestion, persistent indexing, hybrid retrieval, optional reranking, cited outputs, and evaluation artifacts.

If you want one line that captures its value, it is this:

> From raw enterprise documents to cited answers, this tool delivers a practical, inspectable, and extensible RAG workflow.
