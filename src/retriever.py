"""
C2: TF-IDF Retriever

Pure-Python TF-IDF implementation with smoothed IDF and cosine
similarity ranking. No external dependencies.

TF-IDF formula (Equation 1 in the paper):
    tfidf(t, d) = (f_td / |d|) * (log((N+1) / (n_t+1)) + 1)

where f_td is raw term count, |d| is document length in tokens,
N is total chunks, and n_t is chunk frequency of term t.
"""

import math
import re
from collections import Counter


def tokenize(text):
    """Lowercase tokenization on word boundaries."""
    return re.findall(r"\w+", text.lower())


def build_index(chunks):
    """
    Build TF-IDF index over a list of chunks.

    Args:
        chunks: List of chunk dicts (must have 'text' key).

    Returns:
        vectors: List of TF-IDF vector dicts (term -> weight).
        df: Counter of document frequencies.
        N: Number of chunks.
    """
    N = len(chunks)
    df = Counter()
    chunk_token_lists = []

    for chunk in chunks:
        toks = tokenize(chunk["text"])
        chunk_token_lists.append(toks)
        for term in set(toks):
            df[term] += 1

    vectors = []
    for toks in chunk_token_lists:
        tf = Counter(toks)
        doc_len = len(toks)
        vec = {}
        for term, count in tf.items():
            tfidf = (count / doc_len) * (math.log((N + 1) / (df[term] + 1)) + 1)
            vec[term] = tfidf
        vectors.append(vec)

    return vectors, df, N


def cosine_similarity(vec_a, vec_b):
    """Cosine similarity between two sparse vectors (dicts)."""
    common = set(vec_a.keys()) & set(vec_b.keys())
    if not common:
        return 0.0

    dot = sum(vec_a[k] * vec_b[k] for k in common)
    norm_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    norm_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def retrieve(query, chunks, vectors, df, N, top_k=5):
    """
    Retrieve top-k chunks by TF-IDF cosine similarity to query.

    Args:
        query: Query string.
        chunks: List of chunk dicts.
        vectors: TF-IDF vectors from build_index.
        df: Document frequencies from build_index.
        N: Number of chunks.
        top_k: Number of chunks to return.

    Returns:
        List of (similarity_score, chunk_index) tuples, sorted descending.
    """
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

    scores.sort(reverse=True)
    return scores[: min(top_k, len(scores))]
