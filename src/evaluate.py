"""
Evaluation on SQuAD v2.0.

Supports paragraph-level and article-level evaluation modes,
plus ablation studies over scoring weights, refusal thresholds,
and chunk sizes.

Usage:
    python -m src.evaluate --data data/dev-v2.0.json
    python -m src.evaluate --data data/dev-v2.0.json --ablation threshold
    python -m src.evaluate --data data/dev-v2.0.json --ablation weights
    python -m src.evaluate --data data/dev-v2.0.json --ablation chunksize
"""

import json
import time
import argparse
import sys

from .chunker import chunk_document
from .retriever import build_index
from .qa_engine import extractive_qa
from .metrics import exact_match, answer_containment, token_f1


def load_squad(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate(data, theta=0.0, W=200, O=40, alpha=0.5, beta=0.3, gamma=0.2,
             mode="paragraph", verbose=True):
    """
    Run evaluation on SQuAD v2.0 data.

    Args:
        data: Parsed SQuAD JSON.
        theta: Refusal threshold.
        W: Chunk window size.
        O: Chunk overlap.
        alpha, beta, gamma: Scoring weights.
        mode: 'paragraph' (each question uses its paragraph context)
              or 'article' (each article is one document).
        verbose: Print progress.

    Returns:
        Dict of evaluation results.
    """
    total = 0
    answerable_n = 0
    unanswerable_n = 0
    em_correct = 0
    contain_correct = 0
    f1_sum = 0.0
    refusal_correct = 0
    latencies = []

    total_articles = len(data["data"])

    for ai, article in enumerate(data["data"]):
        if mode == "article":
            # Merge all paragraphs into one document
            full_text = " ".join(p["context"] for p in article["paragraphs"])
            chunks = chunk_document(full_text, W, O)
            if not chunks:
                continue
            vectors, df, N = build_index(chunks)

        for para in article["paragraphs"]:
            if mode == "paragraph":
                chunks = chunk_document(para["context"], W, O)
                if not chunks:
                    continue
                vectors, df, N = build_index(chunks)

            for qa in para["qas"]:
                total += 1
                query = qa["question"]
                is_impossible = qa.get("is_impossible", False)

                t0 = time.perf_counter()
                answer, confidence = extractive_qa(
                    query, chunks, vectors, df, N,
                    theta=theta, alpha=alpha, beta=beta, gamma=gamma,
                )
                latency = (time.perf_counter() - t0) * 1000
                latencies.append(latency)

                if is_impossible:
                    unanswerable_n += 1
                    if answer is None:
                        refusal_correct += 1
                else:
                    answerable_n += 1
                    gold_answers = list(set(a["text"] for a in qa["answers"]))
                    if gold_answers:
                        if exact_match(answer, gold_answers):
                            em_correct += 1
                        if answer_containment(answer, gold_answers):
                            contain_correct += 1
                        f1_sum += token_f1(answer, gold_answers)

        if verbose and (ai + 1) % 5 == 0:
            print(f"  {ai+1}/{total_articles} articles, {total} questions...",
                  file=sys.stderr)

    latencies.sort()
    results = {
        "total_questions": total,
        "answerable": answerable_n,
        "unanswerable": unanswerable_n,
        "exact_match": em_correct / answerable_n if answerable_n > 0 else 0,
        "answer_containment": contain_correct / answerable_n if answerable_n > 0 else 0,
        "avg_token_f1": f1_sum / answerable_n if answerable_n > 0 else 0,
        "refusal_accuracy": refusal_correct / unanswerable_n if unanswerable_n > 0 else 0,
        "mean_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
        "p95_latency_ms": latencies[int(len(latencies) * 0.95)] if latencies else 0,
        "max_latency_ms": max(latencies) if latencies else 0,
        "params": {"theta": theta, "W": W, "O": O,
                    "alpha": alpha, "beta": beta, "gamma": gamma},
    }
    return results


def print_results(results, label=""):
    if label:
        print(f"\n{'=' * 60}")
        print(label)
        print("=" * 60)
    print(f"  Total questions:       {results['total_questions']}")
    print(f"  Answerable:            {results['answerable']}")
    print(f"  Unanswerable:          {results['unanswerable']}")
    print(f"  Answer Containment:    {results['answer_containment']:.4f}")
    print(f"  Average Token F1:      {results['avg_token_f1']:.4f}")
    print(f"  Exact Match:           {results['exact_match']:.4f}")
    print(f"  Refusal Accuracy:      {results['refusal_accuracy']:.4f}")
    print(f"  Mean Latency:          {results['mean_latency_ms']:.3f} ms")
    print(f"  p95 Latency:           {results['p95_latency_ms']:.3f} ms")


def main():
    parser = argparse.ArgumentParser(description="Evaluate on SQuAD v2.0")
    parser.add_argument("--data", required=True, help="Path to SQuAD v2.0 JSON")
    parser.add_argument("--mode", default="paragraph",
                        choices=["paragraph", "article"])
    parser.add_argument("--theta", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument("--window-size", type=int, default=200)
    parser.add_argument("--overlap", type=int, default=40)
    parser.add_argument("--ablation", choices=["threshold", "weights", "chunksize"],
                        help="Run ablation study")
    args = parser.parse_args()

    data = load_squad(args.data)
    print(f"Loaded {sum(len(p['qas']) for a in data['data'] for p in a['paragraphs'])} questions")

    if args.ablation == "threshold":
        print("\nRefusal Threshold Ablation")
        print("-" * 70)
        print(f"  {'theta':>5s} | {'Containment':>11s} | {'Token F1':>8s} | {'Refusal':>7s}")
        print("-" * 70)
        for theta in [0.0, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
            r = evaluate(data, theta=theta, W=args.window_size, O=args.overlap,
                         alpha=args.alpha, beta=args.beta, gamma=args.gamma,
                         mode=args.mode, verbose=False)
            print(f"  {theta:5.2f} | {r['answer_containment']:11.4f} | "
                  f"{r['avg_token_f1']:8.4f} | {r['refusal_accuracy']:7.4f}")

    elif args.ablation == "weights":
        print("\nScoring Weight Ablation")
        print("-" * 70)
        configs = [
            ("Default (0.5, 0.3, 0.2)", 0.5, 0.3, 0.2),
            ("Coverage-only (1.0, 0.0, 0.0)", 1.0, 0.0, 0.0),
            ("Density-heavy (0.2, 0.6, 0.2)", 0.2, 0.6, 0.2),
            ("Equal (0.33, 0.33, 0.33)", 0.33, 0.33, 0.33),
            ("No length (0.6, 0.4, 0.0)", 0.6, 0.4, 0.0),
        ]
        for name, a, b, g in configs:
            r = evaluate(data, alpha=a, beta=b, gamma=g,
                         W=args.window_size, O=args.overlap,
                         mode=args.mode, verbose=False)
            print(f"  {name:35s} | Contain={r['answer_containment']:.4f} | "
                  f"F1={r['avg_token_f1']:.4f}")

    elif args.ablation == "chunksize":
        print("\nChunk Size Ablation")
        print("-" * 70)
        for W in [50, 100, 150, 200, 300, 400]:
            O = W // 5
            r = evaluate(data, W=W, O=O, alpha=args.alpha, beta=args.beta,
                         gamma=args.gamma, mode=args.mode, verbose=False)
            print(f"  W={W:3d}, O={O:2d} | Contain={r['answer_containment']:.4f} | "
                  f"F1={r['avg_token_f1']:.4f} | p95={r['p95_latency_ms']:.3f}ms")

    else:
        r = evaluate(data, theta=args.theta, W=args.window_size, O=args.overlap,
                     alpha=args.alpha, beta=args.beta, gamma=args.gamma,
                     mode=args.mode)
        print_results(r, f"Results (mode={args.mode}, theta={args.theta})")


if __name__ == "__main__":
    main()
