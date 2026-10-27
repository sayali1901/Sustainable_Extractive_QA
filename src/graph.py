"""Build term links between document chunks."""

import re
from collections import Counter, defaultdict


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "in", "is", "it", "of", "on", "or", "that", "the", "to",
    "was", "were", "with", "this", "these", "those", "into", "than",
    "their", "its", "can", "will", "would", "should", "using", "use",
}


def keywords(text, limit=8):
    """Return the most common keywords."""
    # Ignore short words and common filler terms.
    terms = [
        t.lower()
        for t in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text)
        if t.lower() not in STOPWORDS
    ]
    counts = Counter(terms)
    # Alphabetical order resolves equal counts.
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [term for term, _ in ranked[:limit]]


def build_document_graph(chunks, max_terms=12):
    """Map common terms to chunk IDs."""
    # Limit links so prompt hints stay small.
    term_chunks = defaultdict(list)
    term_counts = Counter()

    for chunk in chunks:
        for term in keywords(chunk.get("text", ""), limit=10):
            term_counts[term] += 1
            # Four links are enough for a compact hint.
            if len(term_chunks[term]) < 4:
                term_chunks[term].append(chunk.get("id", "unknown"))

    selected_terms = [
        term for term, _ in sorted(term_counts.items(),
                                   key=lambda item: (-item[1], item[0]))[:max_terms]
    ]
    return {
        "nodes": selected_terms,
        "edges": {term: term_chunks[term] for term in selected_terms},
    }


def format_graph(graph):
    """Format graph hints for a prompt."""
    # Give the prompt a clear empty state.
    if not graph or not graph.get("nodes"):
        return "No graph hints."
    lines = []
    for term in graph["nodes"]:
        # One line represents one term and its chunks.
        chunks = ", ".join(graph["edges"].get(term, []))
        lines.append(f"- {term}: {chunks}")
    return "\n".join(lines)
