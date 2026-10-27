# Sustainable Extractive QA

This project answers questions from local documents without sending their contents to an external service. It combines TF-IDF retrieval, extractive question answering, and an optional Ollama model. If no local model is available, it returns the best sentence from the source text.

The core pipeline uses only the Python standard library and runs on a CPU.

## Results

The final configuration was evaluated on all 5,928 answerable questions in the SQuAD v2.0 development set.

| Setting | Answer containment |
|---|---:|
| Extractive fallback, 100-question test | 0.73 |
| Local RAG, 100-question test | 0.87 |
| Local RAG, full answerable set | 0.811 |
| GPT-4o-mini baseline, 100-question test | 0.98 |

The 100-question runs were used to choose the model, prompt, retrieval count, and latency budget. The full-set run was then used as the final evaluation. Detailed results and raw outputs are in [`results/`](results/).

## How it works

1. Documents are split into overlapping chunks.
2. A TF-IDF index ranks chunks against the question.
3. The extractive path scores sentences using query coverage, density, length, and retrieval relevance.
4. If Ollama is available, the retrieved context is passed to a local model with citations.
5. If generation is unavailable or times out, the extractive answer is returned.

Each request records its latency, confidence, citations, model, prompt version, and fallback reason.

## Requirements

- Python 3.9 or newer
- Optional: [Ollama](https://ollama.com/) and a local model such as `qwen3:1.7b`
- Optional: the `openai` Python package for recreating the closed-model baseline

No Python packages are required for the extractive pipeline, evaluation tools, or web interface.

## Setup

Download the SQuAD v2.0 development set:

```bash
curl -L https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json \
  -o data/dev-v2.0.json
```

The dataset is not tracked because it is about 10 MB. See [`data/README.md`](data/README.md) for its source and citation.

## Use the extractive pipeline

Ask one question:

```bash
python -m src.pipeline \
  --document path/to/document.txt \
  --query "What is the main finding?"
```

Start an interactive session:

```bash
python -m src.pipeline --document path/to/document.txt --interactive
```

An audit log is written to `audit_log.jsonl` unless another path is supplied with `--audit-log`.

## Use the local web interface

Start the interface from the project directory:

```bash
python -m src.ui --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` in a browser. The interface searches the local SQuAD file and uses Ollama when a model is running. Otherwise, it uses the extractive fallback.

The question box starts empty. Five example buttons are integrated directly below it; selecting one copies its complete question into the box, where it can be edited before choosing **Ask** or **Summarize evidence**. This keeps the examples available without preselecting or automatically submitting a question.

### Sample questions

1. What does a U.S. census report state that even after other factors there still exists this between earnings of men and women?
2. Who is the author of the book, "Knowledge and Decisions"?
3. What is Southern California often abbreviated as?
4. What is the atomic number of oxygen?
5. When did the 1973 oil crisis begin?

The experiment panel exposes the settings used in the evaluation: local model, prompt, retrieved chunks, chunk window, overlap, latency budget, context budget, generation, and self-correction. Indexes are cached by chunk configuration so repeated comparisons reuse the same retrieval work.

To select a model or Ollama endpoint:

```bash
export LOCAL_QA_MODEL="qwen3:1.7b"
export LOCAL_QA_OLLAMA_ENDPOINT="http://127.0.0.1:11434"
```

## Evaluation

Run the extractive evaluation:

```bash
python -m src.evaluate --data data/dev-v2.0.json --mode paragraph
```

Run an ablation study:

```bash
python -m src.evaluate --data data/dev-v2.0.json --ablation threshold
python -m src.evaluate --data data/dev-v2.0.json --ablation weights
python -m src.evaluate --data data/dev-v2.0.json --ablation chunksize
```

Benchmark the local RAG configuration:

```bash
python -m src.benchmark_variant3 \
  --data data/dev-v2.0.json \
  --limit 100 \
  --answerable-only \
  --use-generator \
  --model qwen3:1.7b \
  --top-k 3 \
  --window-size 100 \
  --overlap 20 \
  --latency-target-ms 5000 \
  --max-context-tokens 4000
```

## Optional closed-model baseline

The closed-model script is separate from the local runtime. Install its dependency and set your API key only if you need to recreate that comparison:

```bash
pip install openai
export OPENAI_API_KEY="your-key"
python openai_baseline.py \
  --data data/dev-v2.0.json \
  --limit 100 \
  --output results/closed_baseline_openai.csv
```

Score saved baseline predictions with `--closed-baseline-csv` when running `src.benchmark_variant3`.

## Project layout

```text
src/
  chunker.py             document chunking
  retriever.py           TF-IDF indexing and retrieval
  qa_engine.py           extractive sentence scoring
  rag_pipeline.py        local generation and fallback logic
  local_llm.py           Ollama client
  prompts.py             prompt templates and context limits
  graph.py               optional term-to-chunk hints
  audit.py               JSONL audit records
  metrics.py             evaluation metrics
  evaluate.py            extractive SQuAD evaluation
  benchmark_variant3.py  local RAG benchmark
  pipeline.py            extractive command-line interface
  ui.py                  local web interface
results/                 evaluation tables and raw outputs
data/                    dataset instructions
openai_baseline.py       optional closed-model comparison
```

## Reproducibility

- Retrieval and extractive scoring are deterministic.
- Local generation uses temperature `0` and seed `42`.
- Prompt templates have a fixed version string.
- Raw experiment outputs are stored as UTF-8 JSON in `results/`.
- Hardware load and local model versions can still affect latency and generated wording.
