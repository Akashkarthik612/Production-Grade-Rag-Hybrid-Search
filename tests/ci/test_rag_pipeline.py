from src.rag.generation.cite_answer import build_query_report
from src.rag.retrieval.rerank import rerank_with_cohere
from src.rag.retrieval.retrieve import reciprocal_rank_fusion


def test_reciprocal_rank_fusion_tracks_modalities() -> None:
    bm25_results = [
        {"id": "a", "document": "Alpha", "metadata": {"chunk_index": 0}, "bm25_score": 8.0},
        {"id": "b", "document": "Beta", "metadata": {"chunk_index": 1}, "bm25_score": 7.0},
    ]
    vector_results = [
        {"id": "a", "document": "Alpha", "metadata": {"chunk_index": 0}, "distance": 0.1},
        {"id": "c", "document": "Gamma", "metadata": {"chunk_index": 2}, "distance": 0.2},
    ]

    fused = reciprocal_rank_fusion(bm25_results, vector_results, top_k=3, rrf_k=60)

    assert fused[0]["id"] == "a"
    assert fused[0]["matched_by"] == "bm25+vector"
    assert fused[1]["matched_by"] in {"bm25", "vector"}


def test_build_query_report_uses_top_result_for_answer_and_citation() -> None:
    results = [
        {
            "document": "Transformers use self-attention to mix token information across a sequence.",
            "metadata": {"filename": "paper.pdf", "chunk_index": 4},
            "matched_by": "bm25+vector",
            "rrf_score": 0.9,
        }
    ]

    report = build_query_report("How do transformers work?", results)

    assert "self-attention" in report["answer"]
    assert report["citations"][0].startswith("[1] paper.pdf chunk 4")


def test_rerank_with_cohere_skips_without_api_key() -> None:
    candidates = [{"document": "doc one", "metadata": {}}]

    reranked, status = rerank_with_cohere(
        query="test",
        candidates=candidates,
        api_key="",
    )

    assert reranked == candidates
    assert status["status"] == "skipped"
