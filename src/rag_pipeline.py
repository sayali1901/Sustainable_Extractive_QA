"""Local RAG with an extractive fallback."""

import re
import time

from .audit import create_audit_entry
from .chunker import chunk_document
from .graph import build_document_graph, format_graph
from .local_llm import LocalLLM
from .prompts import (
    PROMPT_PACK_VERSION,
    estimate_tokens,
    format_context_entries,
    render_prompt,
)
from .qa_engine import rank_answer_candidates, split_sentences
from .retriever import build_index, retrieve


YEAR_RE = re.compile(r"\b([1-9][0-9]{2,3})\b")

# Let CPU generation finish by default.
UNLIMITED_BUDGET_MS = 60000


def _sentence_case(text):
    # Capitalize without changing the remaining text.
    text = text.strip()
    if not text:
        return text
    return text[0].upper() + text[1:]


def compress_extractive_answer(query, answer):
    """Shorten fallback answers to simple date questions."""
    # Only shorten answers that contain a year.
    if not answer or answer == "Insufficient evidence.":
        return answer

    year_match = YEAR_RE.search(answer)
    if not year_match:
        return answer

    year = year_match.group(1)
    query_clean = query.strip().rstrip("?")
    # Preserve the question subject in the short answer.
    begin_match = re.match(r"(?i)^when\s+did\s+(.+?)\s+begin$", query_clean)
    if begin_match:
        subject = _sentence_case(begin_match.group(1))
        return f"{subject} began in {year}."

    happen_match = re.match(
        r"(?i)^when\s+did\s+(.+?)\s+(?:happen|occur|take place)$",
        query_clean,
    )
    if happen_match:
        subject = _sentence_case(happen_match.group(1))
        return f"{subject} happened in {year}."

    if re.search(r"(?i)\b(when|what year|which year)\b", query):
        return year

    return answer


class LocalRAGPipeline:
    """Answer from local documents and models."""

    def __init__(self, window_size=400, overlap=80, theta=0.30,
                 alpha=0.5, beta=0.3, gamma=0.2, top_k=5,
                 generator=None):
        # Keep retrieval and generation settings together.
        self.window_size = window_size
        self.overlap = overlap
        self.theta = theta
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.top_k = top_k
        self.generator = generator or LocalLLM()

        self.chunks = []
        self.vectors = []
        self.df = None
        self.N = 0
        self.audit_entries = []

    def load_texts(self, documents):
        """Index a list of (name, text) pairs."""
        # Prefix chunk IDs with their document ID.
        self.chunks = []
        for doc_index, (name, text) in enumerate(documents):
            doc_id = f"D{doc_index + 1}"
            for chunk in chunk_document(text, self.window_size, self.overlap):
                # Keep local and globally unique chunk IDs.
                enriched = dict(chunk)
                enriched["local_chunk_id"] = chunk["id"]
                enriched["id"] = f"{doc_id}:{chunk['id']}"
                enriched["doc_id"] = doc_id
                enriched["source"] = name
                self.chunks.append(enriched)

        # Empty input clears the previous index.
        if self.chunks:
            self.vectors, self.df, self.N = build_index(self.chunks)
        else:
            self.vectors, self.df, self.N = [], None, 0
        return len(self.chunks)

    def load_files(self, paths):
        """Index UTF-8 text files."""
        # Preserve each path as its source label.
        documents = []
        for path in paths:
            with open(path, "r", encoding="utf-8") as handle:
                documents.append((path, handle.read()))
        return self.load_texts(documents)

    def _retrieved_context_entries(self, query):
        # Add short labels for prompt citations.
        retrieved = retrieve(
            query, self.chunks, self.vectors, self.df, self.N, top_k=self.top_k
        )
        entries = []
        for label_index, (score, chunk_index) in enumerate(retrieved, start=1):
            chunk = self.chunks[chunk_index]
            entries.append({
                "label": f"C{label_index}",
                "score": score,
                "chunk_id": chunk["id"],
                "source": chunk.get("source", "document"),
                "text": chunk["text"],
            })
        return entries

    def _extractive_answer(self, query):
        # Use phrase bonuses only in the RAG fallback.
        candidates = rank_answer_candidates(
            query, self.chunks, self.vectors, self.df, self.N,
            alpha=self.alpha, beta=self.beta, gamma=self.gamma,
            top_k=self.top_k, use_query_bonuses=True,
        )
        if not candidates:
            return "Insufficient evidence.", 0.0, []

        best = candidates[0]
        # Weak evidence should not produce an answer.
        if best["score"] < self.theta:
            return "Insufficient evidence.", best["score"], []

        citations = [best["chunk_id"]]
        return best["sentence"], best["score"], citations

    def _extractive_summary(self, context_entries, max_sentences=3):
        # Take one sentence from each source chunk.
        sentences = []
        citations = []
        for entry in context_entries:
            for sentence in split_sentences(entry["text"]):
                # Skip repeats from overlapping chunks.
                if sentence not in sentences:
                    sentences.append(sentence)
                    citations.append(entry["chunk_id"])
                    break
            if len(sentences) >= max_sentences:
                break
        if not sentences:
            return "Insufficient evidence.", []
        return "\n".join(f"- {sentence}" for sentence in sentences), citations

    def answer(self, query, mode="answer", prompt_id="rag_cited_answer",
               use_generator=True, self_correct=False,
               latency_target_ms=UNLIMITED_BUDGET_MS,
               max_context_tokens=900, max_new_tokens=80, model=None):
        """Answer a question or summarize the loaded text."""
        # Fail clearly when there is nothing to search.
        if not self.chunks:
            raise RuntimeError("No documents loaded.")

        if model:
            self.generator.model = model

        query = (query or "").strip()
        if mode == "summary" and not query:
            query = "Summarize the document."

        # Retrieval time counts towards the request budget.
        t0 = time.perf_counter()
        context_entries = self._retrieved_context_entries(query)
        context, used_entries, context_tokens = format_context_entries(
            context_entries, max_context_tokens=max_context_tokens
        )
        retrieval_latency_ms = (time.perf_counter() - t0) * 1000

        # Prepare a fallback before calling the model.
        if mode == "summary":
            fallback_answer, fallback_citations = self._extractive_summary(used_entries)
            fallback_confidence = 1.0 if fallback_citations else 0.0
            prompt_id = "short_summary"
        else:
            fallback_answer, fallback_confidence, fallback_citations = (
                self._extractive_answer(query)
            )
            fallback_answer = compress_extractive_answer(query, fallback_answer)

        graph_text = ""
        # Build graph hints only when requested.
        if prompt_id == "graph_answer":
            graph = build_document_graph(used_entries)
            graph_text = format_graph(graph)

        prompt = render_prompt(
            prompt_id, question=query, context=context, graph=graph_text
        )
        prompt_tokens_est = estimate_tokens(prompt)
        fallback_reason = ""
        generated = False
        generation_latency_ms = 0.0
        output_tokens = estimate_tokens(fallback_answer)
        answer = fallback_answer
        generator_result = None

        # Give generation only the time left after retrieval.
        remaining_s = max((latency_target_ms - retrieval_latency_ms) / 1000, 0.0)
        if not use_generator:
            fallback_reason = "generator_disabled"
        elif remaining_s < 0.2:
            fallback_reason = "latency_target_exceeded_before_generation"
        else:
            # Fixed sampling keeps runs repeatable.
            generator_result = self.generator.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                timeout_s=remaining_s,
                temperature=0.0,
                top_p=1.0,
                num_ctx=max_context_tokens + max_new_tokens + 256,
            )
            generation_latency_ms = generator_result.latency_ms
            if generator_result.ok:
                answer = generator_result.text
                generated = True
                output_tokens = generator_result.output_tokens

                if self_correct:
                    # Correction uses the remaining request time.
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    correction_budget_s = max(
                        (latency_target_ms - elapsed_ms) / 1000, 0.0
                    )
                    if correction_budget_s >= 0.2:
                        correction_prompt = render_prompt(
                            "self_correct",
                            question=query,
                            context=context,
                            draft_answer=answer,
                        )
                        correction = self.generator.generate(
                            correction_prompt,
                            max_new_tokens=max_new_tokens,
                            timeout_s=correction_budget_s,
                            temperature=0.0,
                            top_p=1.0,
                            num_ctx=max_context_tokens + max_new_tokens + 256,
                        )
                        generation_latency_ms += correction.latency_ms
                        # Keep the draft if correction fails.
                        if correction.ok:
                            answer = correction.text
                            output_tokens += correction.output_tokens
            else:
                fallback_reason = generator_result.error or "generation_failed"

        total_latency_ms = (time.perf_counter() - t0) * 1000
        # Audit every context chunk sent to the model.
        citation_ids = [entry["chunk_id"] for entry in used_entries]
        selected_model = generator_result.model if generator_result else (model or self.generator.model or "")

        result = {
            "answer": answer,
            "mode": mode,
            "generated": generated,
            "fallback_reason": fallback_reason,
            "confidence": round(fallback_confidence, 6),
            "citations": used_entries,
            "selected_citation_ids": fallback_citations,
            "latency_ms": round(total_latency_ms, 4),
            "retrieval_latency_ms": round(retrieval_latency_ms, 4),
            "generation_latency_ms": round(generation_latency_ms, 4),
            "latency_target_ms": latency_target_ms,
            "max_context_tokens": max_context_tokens,
            "top_k": self.top_k,
            "window_size": self.window_size,
            "overlap": self.overlap,
            "self_correct": self_correct,
            "prompt_id": prompt_id,
            "prompt_pack_version": PROMPT_PACK_VERSION,
            "prompt_tokens": prompt_tokens_est,
            "context_tokens": context_tokens,
            "output_tokens": output_tokens,
            "model": selected_model or "",
            "backend": "ollama",
            "deterministic": True,
        }

        # Keep result and audit metadata aligned.
        audit = create_audit_entry(
            query=query,
            answer=answer,
            confidence=fallback_confidence,
            latency_ms=total_latency_ms,
            chunk_ids=citation_ids,
            metadata={
                "mode": mode,
                "generated": generated,
                "fallback_reason": fallback_reason,
                "retrieval_latency_ms": round(retrieval_latency_ms, 4),
                "generation_latency_ms": round(generation_latency_ms, 4),
                "latency_target_ms": latency_target_ms,
                "prompt_id": prompt_id,
                "prompt_pack_version": PROMPT_PACK_VERSION,
                "prompt_tokens": prompt_tokens_est,
                "context_tokens": context_tokens,
                "output_tokens": output_tokens,
                "model": selected_model or "",
                "backend": "ollama",
            },
        )
        self.audit_entries.append(audit)
        return result
