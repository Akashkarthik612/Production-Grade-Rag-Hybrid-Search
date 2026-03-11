"""Local document ingestion using os-based directory traversal."""

import hashlib
import os
from pathlib import Path
from typing import Any
import pypdf


def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        return handle.read().strip()


def _read_pdf_file(path: str) -> str:
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PDF support requires 'pypdf'. Install it with: pip install pypdf"
        ) from exc

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


def _make_doc_id(path: str) -> str:
    rel = path.replace("\\", "/").lower()
    return hashlib.sha1(rel.encode("utf-8")).hexdigest()[:20]


def load_documents_from_local(
    papers_dir: str,
    allowed_extensions: tuple[str, ...] = (".txt", ".md", ".pdf"),
) -> list[dict[str, Any]]:
    """Load local research papers into a standard document list format."""
    if not os.path.isdir(papers_dir):
        raise FileNotFoundError(f"papers_dir not found: {papers_dir}")

    documents: list[dict[str, Any]] = []
    for root, _dirs, files in os.walk(papers_dir):
        for file_name in files:
            ext = Path(file_name).suffix.lower()
            if ext not in allowed_extensions:
                continue

            file_path = os.path.join(root, file_name)
            if ext in (".txt", ".md"):
                text = _read_text_file(file_path)
            elif ext == ".pdf":
                text = _read_pdf_file(file_path)
            else:
                continue

            if not text:
                continue

            doc_id = _make_doc_id(file_path)
            documents.append(
                {
                    "id": doc_id,
                    "text": text,
                    "source": file_path,
                    "metadata": {
                        "filename": file_name,
                        "file_extension": ext,
                    },
                }
            )

    return documents
