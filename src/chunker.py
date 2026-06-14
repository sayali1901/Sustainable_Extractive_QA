"""
C1: Document Chunker

Fixed-window segmentation with configurable window size and overlap.
Each chunk receives a deterministic MD5 identifier from its content
and positional index, ensuring reproducibility.
"""

import hashlib


def chunk_document(text, window_size=200, overlap=40):
    """
    Segment a document into overlapping chunks.

    Args:
        text: Input document as a string.
        window_size: Number of tokens per chunk (default: 200).
        overlap: Number of overlapping tokens between chunks (default: 40).

    Returns:
        List of chunk dicts with keys: id, text, index.
    """
    tokens = text.split()
    chunks = []
    start = 0
    idx = 0

    while start < len(tokens):
        end = min(start + window_size, len(tokens))
        chunk_text = " ".join(tokens[start:end])

        # Deterministic MD5 identifier from content + position
        content_hash = hashlib.md5(
            f"{chunk_text}:{idx}".encode("utf-8")
        ).hexdigest()[:12]

        chunks.append({
            "id": content_hash,
            "text": chunk_text,
            "index": idx,
            "start_token": start,
            "end_token": end,
        })

        idx += 1
        start += window_size - overlap

    return chunks
