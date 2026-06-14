"""
Variant 3 benchmarking harness.

Runs the same public corpus through retrieval-only and optional local-LLM RAG
settings, while recording prompt/token/latency/fallback metadata. Closed-LLM
baselines can be compared by supplying saved predictions; this script does not
tune or improve closed models.
"""

import argparse
import csv
import json
import statistics
import time

from .metrics import answer_containment, exact_match, token_f1
from .rag_pipeline import LocalRAGPipeline


def load_squad_items(path, limit=None):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    items = []
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                if qa.get("is_impossible"):
                    gold = []
                else:
                    gold = list({answer["text"] for answer in qa.get("answers", [])})
                items.append({
                    "context": paragraph["context"],
                    "question": qa["question"],
                    "gold": gold,
                    "is_impossible": qa.get("is_impossible", False),
                })
                if limit and len(items) >= limit:
                    return items
    return items


def percentile(values, p):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(len(ordered) * p), len(ordered) - 1)
    return ordered[index]


def evaluate_setting(items, prompt_id, use_generator, latency_target_ms,
                     max_context_tokens, model):
    latencies = []
    prompt_tokens = []
    context_tokens = []
    output_tokens = []
    fallbacks = 0
    generated = 0
    em = 0
    containment = 0
    f1 = 0.0
    answerable = 0
    refusals = 0
    unanswerable = 0

    t0 = time.perf_counter()
    for item in items:
        pipeline = LocalRAGPipeline()
        pipeline.load_texts([("squad_context", item["context"])])
        result = pipeline.answer(
            item["question"],
            prompt_id=prompt_id,
            use_generator=use_generator,
            latency_target_ms=latency_target_ms,
            max_context_tokens=max_context_tokens,
            model=model,
        )
        prediction = result["answer"]
        latencies.append(result["latency_ms"])
        prompt_tokens.append(result["prompt_tokens"])
        context_tokens.append(result["context_tokens"])
        output_tokens.append(result["output_tokens"])
        fallbacks += 0 if result["generated"] else 1
        generated += 1 if result["generated"] else 0

        if item["is_impossible"]:
            unanswerable += 1
            if "insufficient evidence" in prediction.lower():
                refusals += 1
        else:
            answerable += 1
            em += 1 if exact_match(prediction, item["gold"]) else 0
            containment += 1 if answer_containment(prediction, item["gold"]) else 0
            f1 += token_f1(prediction, item["gold"])

    elapsed = (time.perf_counter() - t0) * 1000
    return {
        "items": len(items),
        "prompt_id": prompt_id,
        "model": model or "auto",
        "use_generator": use_generator,
        "latency_target_ms": latency_target_ms,
        "max_context_tokens": max_context_tokens,
        "answerable": answerable,
        "unanswerable": unanswerable,
        "exact_match": em / answerable if answerable else 0,
        "answer_containment": containment / answerable if answerable else 0,
        "token_f1": f1 / answerable if answerable else 0,
        "refusal_accuracy": refusals / unanswerable if unanswerable else 0,
        "mean_latency_ms": statistics.mean(latencies) if latencies else 0,
        "p95_latency_ms": percentile(latencies, 0.95),
        "mean_prompt_tokens": statistics.mean(prompt_tokens) if prompt_tokens else 0,
        "mean_context_tokens": statistics.mean(context_tokens) if context_tokens else 0,
        "mean_output_tokens": statistics.mean(output_tokens) if output_tokens else 0,
        "generated": generated,
        "fallbacks": fallbacks,
        "total_elapsed_ms": elapsed,
    }


def load_closed_baseline(path):
    """Load saved closed-LLM predictions from CSV: question,prediction,gold."""
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
    answerable = len([row for row in rows if row["gold"]])
    if not answerable:
        return {"closed_baseline_items": len(rows)}
    em = sum(1 for row in rows if exact_match(row["prediction"], row["gold"]))
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
    args = parser.parse_args()

    items = load_squad_items(args.data, limit=args.limit)
    result = evaluate_setting(
        items,
        prompt_id=args.prompt_id,
        use_generator=args.use_generator,
        latency_target_ms=args.latency_target_ms,
        max_context_tokens=args.max_context_tokens,
        model=args.model or None,
    )

    if args.closed_baseline_csv:
        result["closed_baseline"] = evaluate_closed_baseline(
            load_closed_baseline(args.closed_baseline_csv)
        )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
