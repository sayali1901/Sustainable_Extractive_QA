"""JSON audit logging."""

import json
import hashlib
import time
import random

# Keep runs repeatable.
SEED = 42
random.seed(SEED)


def generate_run_id(query, timestamp):
    """Build a repeatable run ID."""
    # Use the same inputs for every run ID.
    content = f"{query}:{timestamp}:{SEED}"
    return hashlib.md5(content.encode("utf-8")).hexdigest()[:16]


def create_audit_entry(query, answer, confidence, latency_ms,
                       chunk_ids=None, deterministic=True, metadata=None):
    """Build one audit entry."""
    # Store both machine and readable timestamps.
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
        # Runtime-specific fields extend the base record.
        entry.update(metadata)
    return entry


def write_audit_log(entries, path="audit_log.jsonl"):
    """Append entries to a JSONL file."""
    # Keep one JSON object per line.
    with open(path, "a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
