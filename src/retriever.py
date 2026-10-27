"""Rank chunks with TF-IDF similarity."""

import math
import re
from collections import Counter


def tokenize(text):
    """Split text into lowercase words."""
    # Keep letters, numbers, and underscores.
    return re.findall(r"\w+", text.lower())


def build_index(chunks):
    """Build TF-IDF vectors for the chunks."""
    # Count how many chunks contain each term.
    N = len(chunks)
    df = Counter()
    chunk_token_lists = []

    for chunk in chunks:
        toks = tokenize(chunk["text"])
        chunk_token_lists.append(toks)
        # Count each term once per chunk for IDF.
        for term in set(toks):
            df[term] += 1

    # Convert each chunk into a sparse TF-IDF vector.
    vectors = []
    for toks in chunk_token_lists:
        tf = Counter(toks)
        doc_len = len(toks)
        vec = {}
        for term, count in tf.items():
            # Smoothed IDF avoids division by zero.
            tfidf = (count / doc_len) * (math.log((N + 1) / (df[term] + 1)) + 1)
            vec[term] = tfidf
        vectors.append(vec)

    return vectors, df, N


def cosine_similarity(vec_a, vec_b):
    """Compare two sparse vectors."""
    # Only shared terms affect the dot product.
    common = set(vec_a.keys()) & set(vec_b.keys())
    if not common:
        return 0.0

    dot = sum(vec_a[k] * vec_b[k] for k in common)
    norm_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    norm_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    # Normalize the dot product by both vector lengths.
    return dot / (norm_a * norm_b)


def retrieve(query, chunks, vectors, df, N, top_k=5):
    """Return the closest chunks and their scores."""
    # Build the query vector with the index weights.
    q_toks = tokenize(query)
    q_tf = Counter(q_toks)
    q_len = len(q_toks)

    if q_len == 0:
        return []

    q_vec = {}
    for term, count in q_tf.items():
        q_vec[term] = (count / q_len) * (
            math.log((N + 1) / (df.get(term, 0) + 1)) + 1
        )

    scores = []
    for i, vec in enumerate(vectors):
        sim = cosine_similarity(q_vec, vec)
        scores.append((sim, i))

    # Tuple sorting also keeps ties deterministic.
    scores.sort(reverse=True)
    return scores[: min(top_k, len(scores))]
