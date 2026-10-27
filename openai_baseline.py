"""Create an OpenAI baseline for the local benchmark."""

import argparse
import csv
import json
import os
import sys

PROMPT_TEMPLATE = """You are answering a reading-comprehension question using only the passage below. Give a short, factual answer using words from the passage. Do not add commentary.

Passage:
{context}

Question: {question}

Answer:"""


def load_answerable(path, limit):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    items = []
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                if qa.get("is_impossible"):
                    continue
                gold = list({a["text"] for a in qa.get("answers", [])})
                if not gold:
                    continue
                items.append({
                    "context": paragraph["context"],
                    "question": qa["question"],
                    "gold": gold,
                })
                if limit and len(items) >= limit:
                    return items
    return items


def call_openai(client, model, context, question):
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=80,
        messages=[
            {"role": "user",
             "content": PROMPT_TEMPLATE.format(context=context, question=question)},
        ],
    )
    return response.choices[0].message.content.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/dev-v2.0.json")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--model", default="gpt-4o-mini",
                        help="OpenAI model, e.g. gpt-4o-mini, gpt-4o, gpt-4.1")
    parser.add_argument("--output", default="results/closed_baseline_openai.csv")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    # Keep the optional package out of the local runtime.
    try:
        from openai import OpenAI
    except ImportError:
        print("Install the optional baseline package with: pip install openai",
              file=sys.stderr)
        sys.exit(1)

    client = OpenAI()
    items = load_answerable(args.data, args.limit)
    print(f"Running {len(items)} items through {args.model}...", file=sys.stderr)

    with open(args.output, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["question", "prediction", "gold"])
        writer.writeheader()
        for i, item in enumerate(items, 1):
            try:
                prediction = call_openai(client, args.model,
                                         item["context"], item["question"])
            except Exception as exc:
                print(f"  [{i}] error: {exc}", file=sys.stderr)
                prediction = ""
            writer.writerow({
                "question": item["question"],
                "prediction": prediction,
                "gold": "|".join(item["gold"]),
            })
            if i % 10 == 0:
                print(f"  {i}/{len(items)}", file=sys.stderr)

    print(f"Wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
