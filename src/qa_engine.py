"""Select answers from retrieved sentences."""

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
    """Split text on sentence endings."""
    # Keep punctuation attached to each sentence.
    sents = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sents if s.strip()]


def score_sentence(sentence, query, alpha=0.5, beta=0.3, gamma=0.2):
    """Score a sentence against the query."""
    # Use unique terms for overlap scoring.
    q_toks = set(tokenize(query))
    s_toks = tokenize(sentence)
    s_set = set(s_toks)

    if not q_toks or not s_set:
        return 0.0

    overlap = s_set & q_toks
    # Coverage rewards recall; density rewards focus.
    coverage = len(overlap) / len(q_toks)
    density = len(overlap) / len(s_set)
    # Cap the length reward at 30 words.
    length = min(len(s_toks) / 30, 1.0)

    return alpha * coverage + beta * density + gamma * length


def query_bonus(sentence, query):
    """Boost phrase matches and dated answers."""
    # Drop question words before building phrases.
    q_toks = [tok for tok in tokenize(query) if tok not in STOPWORDS]
    s_norm = " ".join(tokenize(sentence))
    bonus = 0.0

    if len(q_toks) >= 2:
        # Consecutive terms are stronger than isolated matches.
        bigrams = [" ".join(q_toks[i:i + 2]) for i in range(len(q_toks) - 1)]
        if any(phrase in s_norm for phrase in bigrams):
            bonus += 0.14

    if len(q_toks) >= 3:
        trigrams = [" ".join(q_toks[i:i + 3]) for i in range(len(q_toks) - 2)]
        if any(phrase in s_norm for phrase in trigrams):
            bonus += 0.08

    q_words = set(tokenize(query))
    # Prefer dated evidence for date questions.
    if {"when", "year", "date"} & q_words and DATE_RE.search(sentence):
        bonus += 0.12

    return bonus


def rank_answer_candidates(query, chunks, vectors, df, N,
                           alpha=0.5, beta=0.3, gamma=0.2, top_k=5,
                           use_query_bonuses=False):
    """Rank sentences from the retrieved chunks."""
    # Search first, then score each sentence.
    from .retriever import retrieve

    retrieved = retrieve(query, chunks, vectors, df, N, top_k=top_k)
    seen_chunks = set()
    candidates = []

    for ret_score, chunk_idx in retrieved:
        chunk = chunks[chunk_idx]
        sentences = split_sentences(chunk["text"])

        for sent in sentences:
            s_score = score_sentence(sent, query, alpha, beta, gamma)

            # Give each new chunk a small boost.
            eps_div = 0.1 if chunk["id"] not in seen_chunks else 0.0

            # Blend sentence quality with chunk relevance.
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
    """Return the best sentence or refuse a weak match."""
    # The first candidate has the highest score.
    candidates = rank_answer_candidates(
        query, chunks, vectors, df, N,
        alpha=alpha, beta=beta, gamma=gamma, top_k=top_k,
    )

    if not candidates:
        return None, 0.0

    best = candidates[0]

    # Refuse answers below the configured threshold.
    if best["score"] < theta:
        return None, best["score"]

    return best["sentence"], best["score"]
