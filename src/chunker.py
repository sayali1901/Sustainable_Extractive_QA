"""Split documents into overlapping chunks."""

import hashlib


def chunk_document(text, window_size=200, overlap=40):
    """Split text into overlapping word chunks."""
    # Work with words to avoid extra tokenizers.
    tokens = text.split()
    chunks = []
    start = 0
    idx = 0

    while start < len(tokens):
        end = min(start + window_size, len(tokens))
        chunk_text = " ".join(tokens[start:end])

        # Include the position to keep duplicate chunks distinct.
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

        # Move forward while retaining the overlap.
        idx += 1
        start += window_size - overlap

    return chunks
