"""
C3: Extractive QA Engine

Multi-factor sentence scoring with configurable weights and refusal
threshold. Rather than generating answers, the system extracts the
highest-scoring sentence from retrieved chunks.

Scoring function (Equation 2 in the paper):
    score(s) = α * coverage(s,q) + β * density(s,q) + γ * length(s)

Final score (Equation 3):
    final(s) = 0.6 * score(s) + 0.4 * retrieval_score(chunk(s)) + ε_div

Refusal: if max(final(s)) < θ, return "Insufficient evidence."
"""

import re
from .retriever import tokenize


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do",
    "does", "for", "from", "how", "in", "is", "it", "of", "on", "or",
    "the", "to", "was", "were", "what", "when", "where", "which", "who",
    "why", "with",
}

DATE_RE = re.compile(
    r"\b(?:\d{3,4}|january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\b",
    re.IGNORECASE,
)


def split_sentences(text):
    """Split text into sentences on sentence-ending punctuation."""
    sents = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sents if s.strip()]


def score_sentence(sentence, query, alpha=0.5, beta=0.3, gamma=0.2):
    """
    Multi-factor sentence scoring.

    Args:
        sentence: Candidate sentence string.
        query: Query string.
        alpha: Weight for coverage (query term recall).
        beta: Weight for density (query term precision in sentence).
        gamma: Weight for length normalisation.

    Returns:
        Float score in [0, 1].
    """
    q_toks = set(tokenize(query))
    s_toks = tokenize(sentence)
    s_set = set(s_toks)

    if not q_toks or not s_set:
        return 0.0

    overlap = s_set & q_toks
    coverage = len(overlap) / len(q_toks)
    density = len(overlap) / len(s_set)
    length = min(len(s_toks) / 30, 1.0)

    return alpha * coverage + beta * density + gamma * length


def query_bonus(sentence, query):
    """
    Deterministic RAG fallback boost for exact phrases and date questions.

    The original scoring is intentionally simple for the paper baseline. This
    optional bonus is used by the UI/RAG path to avoid returning generic
    same-topic sentences when the query contains a strong phrase such as
    "Norman conquest" or asks for a date.
    """
    q_toks = [tok for tok in tokenize(query) if tok not in STOPWORDS]
    s_norm = " ".join(tokenize(sentence))
    bonus = 0.0

    if len(q_toks) >= 2:
        bigrams = [" ".join(q_toks[i:i + 2]) for i in range(len(q_toks) - 1)]
        if any(phrase in s_norm for phrase in bigrams):
            bonus += 0.14

    if len(q_toks) >= 3:
        trigrams = [" ".join(q_toks[i:i + 3]) for i in range(len(q_toks) - 2)]
        if any(phrase in s_norm for phrase in trigrams):
            bonus += 0.08

    q_words = set(tokenize(query))
    if {"when", "year", "date"} & q_words and DATE_RE.search(sentence):
        bonus += 0.12

    return bonus


def rank_answer_candidates(query, chunks, vectors, df, N,
                           alpha=0.5, beta=0.3, gamma=0.2, top_k=5,
                           use_query_bonuses=False):
    """
    Rank extractive answer candidates over retrieved chunks.

    Args:
        query: Query string.
        chunks: List of chunk dicts.
        vectors: TF-IDF vectors.
        df: Document frequencies.
        N: Number of chunks.
        alpha, beta, gamma: Scoring weights.
        top_k: Number of chunks to retrieve.

    Returns:
        List of candidate dicts sorted by descending score.
    """
    from .retriever import retrieve

    retrieved = retrieve(query, chunks, vectors, df, N, top_k=top_k)
    seen_chunks = set()
    candidates = []

    for ret_score, chunk_idx in retrieved:
        chunk = chunks[chunk_idx]
        sentences = split_sentences(chunk["text"])

        for sent in sentences:
            s_score = score_sentence(sent, query, alpha, beta, gamma)

            # Diversity bonus for first citation from each chunk
            eps_div = 0.1 if chunk["id"] not in seen_chunks else 0.0

            final = 0.6 * s_score + 0.4 * ret_score + eps_div
            if use_query_bonuses:
                final += query_bonus(sent, query)

            candidates.append({
                "sentence": sent,
                "score": final,
                "chunk_id": chunk["id"],
                "chunk_index": chunk["index"],
                "retrieval_score": ret_score,
                "sentence_score": s_score,
            })
            seen_chunks.add(chunk["id"])

    if not candidates:
        return []

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates


def extractive_qa(query, chunks, vectors, df, N, theta=0.0,
                  alpha=0.5, beta=0.3, gamma=0.2, top_k=5):
    """
    Run extractive QA over retrieved chunks.

    Args:
        query: Query string.
        chunks: List of chunk dicts.
        vectors: TF-IDF vectors.
        df: Document frequencies.
        N: Number of chunks.
        theta: Refusal threshold. If best score < theta, refuse.
        alpha, beta, gamma: Scoring weights.
        top_k: Number of chunks to retrieve.

    Returns:
        (answer_sentence, confidence_score) or (None, score) if refused.
    """
    candidates = rank_answer_candidates(
        query, chunks, vectors, df, N,
        alpha=alpha, beta=beta, gamma=gamma, top_k=top_k,
    )

    if not candidates:
        return None, 0.0

    best = candidates[0]

    if best["score"] < theta:
        return None, best["score"]

    return best["sentence"], best["score"]
