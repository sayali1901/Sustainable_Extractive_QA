"""Benchmark local RAG settings."""

import argparse
import csv
import json
import statistics

from .metrics import answer_containment, exact_match, token_f1
from .rag_pipeline import LocalRAGPipeline


def load_squad_items(path, limit=None, answerable_only=False):
    # Flatten articles into benchmark items.
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    items = []
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                is_impossible = qa.get("is_impossible", False)
                # Optional filtering supports answerable-only runs.
                if is_impossible and answerable_only:
                    continue
                if is_impossible:
                    gold = []
                else:
                    gold = list({answer["text"] for answer in qa.get("answers", [])})
                items.append({
                    "context": paragraph["context"],
                    "question": qa["question"],
                    "gold": gold,
                    "is_impossible": is_impossible,
                })
                if limit and len(items) >= limit:
                    return items
    return items


def percentile(values, p):
    # Use the nearest observed value.
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(len(ordered) * p), len(ordered) - 1)
    return ordered[index]


def evaluate_setting(items, prompt_id, use_generator, latency_target_ms,
                     max_context_tokens, model,
                     top_k=5, window_size=400, overlap=80,
                     self_correct=False):
    # Collect only values included in the report.
    latencies = []
    context_tokens = []
    containment = 0
    retrieved_chunks = []

    for item in items:
        # Each item gets a fresh index and audit state.
        pipeline = LocalRAGPipeline(
            window_size=window_size, overlap=overlap, top_k=top_k
        )
        pipeline.load_texts([("squad_context", item["context"])])
        result = pipeline.answer(
            item["question"],
            prompt_id=prompt_id,
            use_generator=use_generator,
            latency_target_ms=latency_target_ms,
            max_context_tokens=max_context_tokens,
            model=model,
            self_correct=self_correct,
        )
        prediction = result["answer"]
        latencies.append(result["latency_ms"])
        context_tokens.append(result["context_tokens"])
        retrieved_chunks.append(len(result.get("citations", [])))

        # Containment suits full-sentence answers.
        containment += 1 if answer_containment(prediction, item["gold"]) else 0

    n = len(items)
    return {
        "items": n,
        "prompt_id": prompt_id,
        "model": model or "auto",
        "use_generator": use_generator,
        "latency_target_ms": latency_target_ms,
        "max_context_tokens": max_context_tokens,
        "top_k": top_k,
        "window_size": window_size,
        "overlap": overlap,
        "self_correct": self_correct,
        "answer_containment": containment / n if n else 0,
        "mean_latency_ms": statistics.mean(latencies) if latencies else 0,
        "p95_latency_ms": percentile(latencies, 0.95),
        "mean_context_tokens": statistics.mean(context_tokens) if context_tokens else 0,
        "mean_retrieved_chunks": statistics.mean(retrieved_chunks) if retrieved_chunks else 0,
    }


def load_closed_baseline(path):
    """Load saved baseline predictions from CSV."""
    # Gold answers are pipe-separated in the CSV.
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            gold = [g.strip() for g in row.get("gold", "").split("|") if g.strip()]
            rows.append({
                "question": row.get("question", ""),
                "prediction": row.get("prediction", ""),
                "gold": gold,
            })
    return rows


def evaluate_closed_baseline(rows):
    # Ignore rows without gold answers in rates.
    answerable = len([row for row in rows if row["gold"]])
    if not answerable:
        return {"closed_baseline_items": len(rows)}
    em = sum(1 for row in rows if exact_match(row["prediction"], row["gold"]))
    # Use the same metrics as the local pipeline.
    containment = sum(
        1 for row in rows if answer_containment(row["prediction"], row["gold"])
    )
    f1 = sum(token_f1(row["prediction"], row["gold"]) for row in rows)
    return {
        "closed_baseline_items": len(rows),
        "closed_exact_match": em / answerable,
        "closed_answer_containment": containment / answerable,
        "closed_token_f1": f1 / answerable,
        "note": "Closed baseline predictions are evaluated as supplied; no prompt or model optimization is performed.",
    }


def main():
    # Expose the main benchmark settings.
    parser = argparse.ArgumentParser(description="Benchmark Variant 3 RAG settings")
    parser.add_argument("--data", required=True, help="SQuAD v2.0 JSON path")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--prompt-id", default="rag_cited_answer",
                        choices=["rag_cited_answer", "rag_strict", "graph_answer"])
    parser.add_argument("--use-generator", action="store_true")
    parser.add_argument("--model", default="")
    parser.add_argument("--latency-target-ms", type=int, default=1500)
    parser.add_argument("--max-context-tokens", type=int, default=900)
    parser.add_argument("--closed-baseline-csv")
    parser.add_argument("--answerable-only", action="store_true",
                        help="Skip is_impossible questions (Variant 3 evaluation)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of retrieved chunks fed to the generator")
    parser.add_argument("--window-size", type=int, default=400,
                        help="Chunker window size in tokens")
    parser.add_argument("--overlap", type=int, default=80,
                        help="Chunker overlap in tokens")
    parser.add_argument("--self-correct", action="store_true",
                        help="Run a second-pass refinement over the first answer")
    args = parser.parse_args()

    items = load_squad_items(args.data, limit=args.limit,
                             answerable_only=args.answerable_only)
    result = evaluate_setting(
        items,
        prompt_id=args.prompt_id,
        use_generator=args.use_generator,
        latency_target_ms=args.latency_target_ms,
        max_context_tokens=args.max_context_tokens,
        model=args.model or None,
        top_k=args.top_k,
        window_size=args.window_size,
        overlap=args.overlap,
        self_correct=args.self_correct,
    )

    if args.closed_baseline_csv:
        # Baseline predictions remain separate in the report.
        result["closed_baseline"] = evaluate_closed_baseline(
            load_closed_baseline(args.closed_baseline_csv)
        )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
