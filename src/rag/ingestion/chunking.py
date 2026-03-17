"""Token-aware chunking utility for ingestion."""

import tiktoken


class _FallbackEncoding:
    """Small offline-safe tokenizer for environments without cached tiktoken data."""

    def encode(self, text: str) -> list[str]:
        return text.split()

    def decode(self, tokens: list[str]) -> str:
        return " ".join(tokens)


def _get_encoding():
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return _FallbackEncoding()


def chunk_text(
    text: str,
    min_tokens: int = 500,
    max_tokens: int = 800,
    overlap: int = 80,
) -> list[str]:
    """Split text into token-based chunks and return decoded chunk texts."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= max_tokens:
        raise ValueError("overlap must be smaller than max_tokens")
    if min_tokens <= 0:
        raise ValueError("min_tokens must be > 0")

    encoding = _get_encoding()
    tokens = encoding.encode(text)
    if not tokens:
        return []

    step = max_tokens - overlap
    token_chunks: list[list[int]] = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        token_chunks.append(tokens[start:end])
        if end == len(tokens):
            break
        start += step

    # Keep chunk sizes near the requested range when the tail is too short.
    if len(token_chunks) > 1 and len(token_chunks[-1]) < min_tokens:
        token_chunks[-2].extend(token_chunks[-1])
        token_chunks.pop()

    return [encoding.decode(chunk) for chunk in token_chunks if chunk]
        
