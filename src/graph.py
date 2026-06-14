"""
Lightweight multi-document graph hints for graph-prompts.

This is deliberately simple and deterministic: terms connect to the document
chunks where they appear. It gives the prompt a compact cross-document map
without adding heavy graph dependencies.
"""

import re
from collections import Counter, defaultdict


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "in", "is", "it", "of", "on", "or", "that", "the", "to",
    "was", "were", "with", "this", "these", "those", "into", "than",
    "their", "its", "can", "will", "would", "should", "using", "use",
}


def keywords(text, limit=8):
    """Return deterministic top keywords for a text span."""
    terms = [
        t.lower()
        for t in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text)
        if t.lower() not in STOPWORDS
    ]
    counts = Counter(terms)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [term for term, _ in ranked[:limit]]


def build_document_graph(chunks, max_terms=12):
    """Build a compact term-to-chunk graph from retrieved chunks."""
    term_chunks = defaultdict(list)
    term_counts = Counter()

    for chunk in chunks:
        for term in keywords(chunk.get("text", ""), limit=10):
            term_counts[term] += 1
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
    if not graph or not graph.get("nodes"):
        return "No graph hints."
    lines = []
    for term in graph["nodes"]:
        chunks = ", ".join(graph["edges"].get(term, []))
        lines.append(f"- {term}: {chunks}")
    return "\n".join(lines)
