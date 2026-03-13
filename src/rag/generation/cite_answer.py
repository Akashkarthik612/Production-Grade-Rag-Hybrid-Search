"""Utilities for writing query/answer reports to a Word document."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any
from xml.sax.saxutils import escape
import zipfile


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def format_citation(result: dict[str, Any], index: int = 1) -> str:
    metadata = result.get("metadata", {}) or {}
    filename = metadata.get("filename") or metadata.get("source") or metadata.get("paper_id") or "unknown"
    chunk_index = metadata.get("chunk_index", "unknown")
    matched_by = result.get("matched_by", "unknown")
    score_parts: list[str] = []
    if "rrf_score" in result:
        score_parts.append(f"rrf={result['rrf_score']:.4f}")
    if "rerank_score" in result:
        score_parts.append(f"rerank={result['rerank_score']:.4f}")
    score_suffix = f" | {' '.join(score_parts)}" if score_parts else ""
    return f"[{index}] {filename} chunk {chunk_index} | matched_by={matched_by}{score_suffix}"


def build_extractive_answer(results: list[dict[str, Any]], max_chars: int = 700) -> str:
    if not results:
        return "No answer could be assembled because no relevant chunks were returned."

    text = _normalize_whitespace(str(results[0].get("document", "")))
    if not text:
        return "Top-ranked chunk was empty."

    sentences = re.split(r"(?<=[.!?])\s+", text)
    answer_parts: list[str] = []
    total_chars = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        projected = total_chars + len(sentence) + (1 if answer_parts else 0)
        if projected > max_chars and answer_parts:
            break
        answer_parts.append(sentence)
        total_chars = projected
        if len(answer_parts) >= 3:
            break

    if answer_parts:
        return " ".join(answer_parts)

    return text[:max_chars].rstrip() + ("..." if len(text) > max_chars else "")


def build_query_report(
    query: str,
    results: list[dict[str, Any]],
    rerank_status: dict[str, Any] | None = None,
    citation_count: int = 2,
) -> dict[str, Any]:
    citations = [
        format_citation(result, index=index)
        for index, result in enumerate(results[:citation_count], start=1)
    ]
    report: dict[str, Any] = {
        "query": query,
        "answer": build_extractive_answer(results),
        "citations": citations,
        "results": results,
    }
    if rerank_status:
        report["rerank_status"] = rerank_status
    return report


def _paragraph_xml(text: str) -> str:
    safe_text = escape(text)
    return (
        "<w:p>"
        "<w:r>"
        f"<w:t xml:space=\"preserve\">{safe_text}</w:t>"
        "</w:r>"
        "</w:p>"
    )


def _build_document_xml(paragraphs: list[str]) -> str:
    body = "".join(_paragraph_xml(paragraph) for paragraph in paragraphs)
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<w:document "
        "xmlns:wpc=\"http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas\" "
        "xmlns:mc=\"http://schemas.openxmlformats.org/markup-compatibility/2006\" "
        "xmlns:o=\"urn:schemas-microsoft-com:office:office\" "
        "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\" "
        "xmlns:m=\"http://schemas.openxmlformats.org/officeDocument/2006/math\" "
        "xmlns:v=\"urn:schemas-microsoft-com:vml\" "
        "xmlns:wp14=\"http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing\" "
        "xmlns:wp=\"http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing\" "
        "xmlns:w10=\"urn:schemas-microsoft-com:office:word\" "
        "xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\" "
        "xmlns:w14=\"http://schemas.microsoft.com/office/word/2010/wordml\" "
        "xmlns:w15=\"http://schemas.microsoft.com/office/word/2012/wordml\" "
        "xmlns:wpg=\"http://schemas.microsoft.com/office/word/2010/wordprocessingGroup\" "
        "xmlns:wpi=\"http://schemas.microsoft.com/office/word/2010/wordprocessingInk\" "
        "xmlns:wne=\"http://schemas.microsoft.com/office/word/2006/wordml\" "
        "xmlns:wps=\"http://schemas.microsoft.com/office/word/2010/wordprocessingShape\" "
        "mc:Ignorable=\"w14 w15 wp14\">"
        f"<w:body>{body}<w:sectPr/></w:body>"
        "</w:document>"
    )


def write_query_results_docx(
    output_path: str | Path,
    reports: list[dict[str, Any]],
    title: str = "RAG Query Report",
) -> Path:
    """Write query/result sections into a minimal .docx file using stdlib only."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    paragraphs = [title, ""]
    for report in reports:
        query = str(report.get("query", "")).strip()
        paragraphs.append(f"Query: {query}")
        paragraphs.append("")

        answer = str(report.get("answer", "")).strip()
        if answer:
            paragraphs.append("Answer:")
            paragraphs.append(answer)
            paragraphs.append("")

        citations = report.get("citations", [])
        if citations:
            paragraphs.append("Citations:")
            for citation in citations:
                paragraphs.append(str(citation))
            paragraphs.append("")

        rerank_status = report.get("rerank_status")
        if isinstance(rerank_status, dict) and rerank_status:
            paragraphs.append(f"Reranker: {rerank_status.get('status', 'unknown')}")
            detail = rerank_status.get("detail")
            if detail:
                paragraphs.append(str(detail))
            paragraphs.append("")

        for index, result in enumerate(report.get("results", []), start=1):
            paragraphs.append(f"Result {index}")
            paragraphs.append(f"Document: {result.get('document', '')}")
            paragraphs.append(f"Metadata: {result.get('metadata', {})}")
            paragraphs.append(f"RRF Score: {result.get('rrf_score', 'N/A')}")
            paragraphs.append(f"BM25 Score: {result.get('bm25_score', 'N/A')}")
            paragraphs.append(f"Vector Distance: {result.get('distance', 'N/A')}")
            paragraphs.append(f"Matched By: {result.get('matched_by', 'unknown')}")
            paragraphs.append(f"Rerank Score: {result.get('rerank_score', 'N/A')}")
            paragraphs.append("")

        paragraphs.append("-" * 80)
        paragraphs.append("")

    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>
"""
    package_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
"""
    document_xml = _build_document_xml(paragraphs)

    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as docx:
        docx.writestr("[Content_Types].xml", content_types)
        docx.writestr("_rels/.rels", package_rels)
        docx.writestr("word/document.xml", document_xml)

    return destination.resolve()
