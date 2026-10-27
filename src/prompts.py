"""Prompt templates for local RAG."""

import re


TOKEN_RE = re.compile(r"\w+|[^\w\s]")


PROMPT_PACK_VERSION = "variant3-2026-06-14"


PROMPT_TEMPLATES = {
    "rag_cited_answer": {
        "task": "answer",
        "description": "Short grounded answer with bracket citations.",
        "template": """System:
You answer questions using only the supplied context. Be concise. /no_think
If the context does not contain enough evidence, say: Insufficient evidence.
Every factual claim must include one or more citations like [C1].

Context:
{context}

Question:
{question}

Answer:""",
    },
    "rag_strict": {
        "task": "answer",
        "description": "Stricter context-only answer for higher precision.",
        "template": """System:
Use only the context below. Do not infer missing facts. /no_think
Return at most three sentences. Cite each sentence with [C#].
If no cited sentence can answer the question, return exactly: Insufficient evidence.

Context:
{context}

Question:
{question}

Answer:""",
    },
    "short_summary": {
        "task": "summary",
        "description": "Short cited summary of the retrieved context.",
        "template": """System:
Summarise only the supplied context in 3 concise bullets. /no_think
Each bullet must include a citation like [C1].

Context:
{context}

Summary:""",
    },
    "self_correct": {
        "task": "repair",
        "description": "Second pass that removes unsupported claims.",
        "template": """System:
Check the draft answer against the context. Remove unsupported claims. /no_think
Keep citations. If the answer is unsupported, return: Insufficient evidence.

Context:
{context}

Question:
{question}

Draft answer:
{draft_answer}

Corrected answer:""",
    },
    "graph_answer": {
        "task": "answer",
        "description": "Answer with retrieved context plus a lightweight document graph.",
        "template": """System:
Answer using the context and graph hints. The graph is only a navigation aid; /no_think
the cited context is the source of truth. Keep the answer short and cite [C#].

Graph hints:
{graph}

Context:
{context}

Question:
{question}

Answer:""",
    },
}


def estimate_tokens(text):
    """Estimate tokens without a tokenizer."""
    # Count words and punctuation separately.
    if not text:
        return 0
    return len(TOKEN_RE.findall(text))


def trim_to_token_budget(text, max_tokens):
    """Trim text to an estimated token limit."""
    # Return early when no trimming is needed.
    if max_tokens <= 0:
        return ""
    tokens = TOKEN_RE.findall(text)
    if len(tokens) <= max_tokens:
        return text
    trimmed = tokens[:max_tokens]
    out = ""
    for tok in trimmed:
        # Add spaces between words, not before punctuation.
        if out and re.match(r"\w", tok) and re.match(r".*\w$", out):
            out += " "
        out += tok
    return out.strip()


def get_prompt_template(prompt_id):
    """Get a prompt template by ID."""
    # Unknown IDs use the standard answer prompt.
    return PROMPT_TEMPLATES.get(prompt_id, PROMPT_TEMPLATES["rag_cited_answer"])


def render_prompt(prompt_id, question, context, graph="", draft_answer=""):
    """Fill a prompt template."""
    # Strip user text before inserting it.
    prompt = get_prompt_template(prompt_id)["template"]
    return prompt.format(
        question=question.strip(),
        context=context.strip(),
        graph=graph.strip() or "No graph hints.",
        draft_answer=draft_answer.strip(),
    )


def format_context_entries(entries, max_context_tokens):
    """Format cited chunks within the token limit."""
    # Add chunks until the shared budget is full.
    used = []
    parts = []
    remaining = max_context_tokens

    for entry in entries:
        # Reserve space for the citation label and source.
        prefix = "[{label}] {source}\n".format(
            label=entry["label"], source=entry.get("source", "document")
        )
        prefix_tokens = estimate_tokens(prefix)
        if remaining <= prefix_tokens:
            break

        body = trim_to_token_budget(entry["text"], remaining - prefix_tokens)
        if not body:
            break

        formatted = prefix + body
        parts.append(formatted)
        used_entry = dict(entry)
        used_entry["text"] = body
        used.append(used_entry)
        # Recalculate after formatting to include the prefix.
        remaining -= estimate_tokens(formatted)

    context = "\n\n".join(parts)
    return context, used, estimate_tokens(context)
