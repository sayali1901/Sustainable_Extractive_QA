"""
C4: Audit and Determinism

Structured JSON audit logging for every pipeline run. Records run ID,
timestamp, query, latency, confidence, citation count, and determinism flag.
Fixed seed (s=42) ensures bitwise reproducibility.
"""

import json
import hashlib
import time
import random

# Fixed seed for deterministic behaviour
SEED = 42
random.seed(SEED)


def generate_run_id(query, timestamp):
    """Generate deterministic run ID from query and timestamp."""
    content = f"{query}:{timestamp}:{SEED}"
    return hashlib.md5(content.encode("utf-8")).hexdigest()[:16]


def create_audit_entry(query, answer, confidence, latency_ms,
                       chunk_ids=None, deterministic=True, metadata=None):
    """
    Create a structured audit log entry.

    Args:
        query: The input query string.
        answer: The extracted answer (or None if refused).
        confidence: The confidence score.
        latency_ms: Query latency in milliseconds.
        chunk_ids: List of chunk IDs that contributed to the answer.
        deterministic: Whether the run was deterministic.
        metadata: Optional extra run metadata.

    Returns:
        Dict representing the audit entry.
    """
    timestamp = time.time()
    entry = {
        "run_id": generate_run_id(query, timestamp),
        "timestamp": timestamp,
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp)),
        "query": query,
        "answer": answer,
        "refused": answer is None,
        "confidence": round(confidence, 6),
        "latency_ms": round(latency_ms, 4),
        "citation_count": len(chunk_ids) if chunk_ids else 0,
        "chunk_ids": chunk_ids or [],
        "deterministic": deterministic,
        "seed": SEED,
        "pipeline_version": "1.0.0",
    }
    if metadata:
        entry.update(metadata)
    return entry


def write_audit_log(entries, path="audit_log.jsonl"):
    """Write audit entries to a JSONL file."""
    with open(path, "a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
