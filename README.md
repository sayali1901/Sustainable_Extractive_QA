# Sustainable Extractive QA for Resource-Constrained Environments

A fully offline, zero-dependency extractive question answering system implemented in pure Python. Designed for deployment on standard laptops without GPU or internet access.

**Paper:** *Sustainable Extractive Question Answering for Resource-Constrained Personalized Document Assistants*  
**Venue:** LLM4Good Workshop, ACM UMAP 2026, Gothenburg, Sweden  
**Author:** Sayali Nitin Doifode, King's College London

## Key Results (SQuAD v2.0 Dev Set — 11,873 Questions)

| Metric | Value |
|---|---|
| Answer Containment | 69.0% |
| Token F1 | 0.171 |
| Mean Latency | 0.097 ms |
| p95 Latency | 0.183 ms |
| Peak Memory | <10 MB |
| Est. Energy/Query | ~0.0015 mJ |
| External Dependencies | **Zero** |

## Overview

The system combines a pure-Python TF-IDF retriever with a multi-factor extractive answer selector. It returns document-grounded answers with sub-millisecond latency and under 10 MB of memory — approximately four orders of magnitude less energy per query than quantized small language models.

```
Documents → C1: Chunker → C2: TF-IDF Retriever → C3: Extractive QA → Answer + Citations
                                                     ↑
                                              C4: Audit Logger (logs every stage)
```

## Quick Start

No installation required. The entire system runs on the Python standard library.

```bash
# Ask a question about a document
python -m src.pipeline --document path/to/document.txt --query "What is the main finding?"

# Run evaluation on SQuAD v2.0
python -m src.evaluate --data data/dev-v2.0.json --mode paragraph

# Run with custom parameters
python -m src.evaluate --data data/dev-v2.0.json --theta 0.30 --alpha 0.5 --beta 0.3 --gamma 0.2
```

## Variant 3: Local RAG + UI

This repository also includes a lightweight local RAG component for CPU-only
on-device QA and short summaries. It keeps the original extractive pipeline as
the deterministic fallback and can use an already-installed local Ollama model
such as Qwen or Llama when available.

```bash
# Launch the local UI
python -m src.ui --host 127.0.0.1 --port 8000

# Benchmark prompt/model settings on SQuAD v2.0
python -m src.benchmark_variant3 --data data/dev-v2.0.json --limit 100

# Try local generation if an Ollama model is already installed
python -m src.benchmark_variant3 --data data/dev-v2.0.json --limit 100 --use-generator --model qwen3:0.6b
```

Variant 3 additions:
- Prompt pack for cited answers, strict RAG, short summaries, self-correction, and graph prompts.
- Optional local generator wrapper using Ollama with fixed seed and deterministic settings.
- Latency-aware fallback to retrieval-only answers when generation is unavailable or exceeds the target.
- Multi-document loading and lightweight graph hints.
- Web UI with no external frontend dependencies.
- Benchmark harness logging accuracy, p95 latency, token counts, model/prompt metadata, and fallback counts.

## Project Structure

```
.
├── src/
│   ├── __init__.py
│   ├── chunker.py          # C1: Document chunker with configurable window/overlap
│   ├── retriever.py         # C2: Pure-Python TF-IDF retriever
│   ├── qa_engine.py         # C3: Multi-factor extractive QA with refusal
│   ├── audit.py             # C4: Structured JSON audit logging
│   ├── pipeline.py          # Combined pipeline + CLI interface
│   ├── metrics.py           # Evaluation metrics (EM, Containment, F1)
│   ├── evaluate.py          # Full evaluation on SQuAD v2.0
│   ├── rag_pipeline.py      # Variant 3 local RAG + fallback orchestration
│   ├── prompts.py           # Versioned prompt pack
│   ├── local_llm.py         # Optional Ollama wrapper
│   ├── model_registry.py    # Candidate local model registry
│   ├── graph.py             # Lightweight document graph hints
│   ├── benchmark_variant3.py# Prompt/model benchmark harness
│   └── ui.py                # Minimal local web UI
├── data/
│   └── README.md            # Instructions for downloading SQuAD v2.0
├── results/
│   └── full_results.md      # Complete evaluation results
├── paper/
│   └── paper.tex            # Camera-ready paper source
├── LICENSE
├── README.md
└── requirements.txt         # Empty — zero dependencies
```

## System Components

### C1: Document Chunker (`src/chunker.py`)
Fixed-window segmentation with configurable window size W and overlap O. Each chunk receives a deterministic MD5 identifier for reproducibility.

### C2: TF-IDF Retriever (`src/retriever.py`)
Pure-Python TF-IDF with smoothed IDF and cosine similarity ranking. Returns top-k chunks.

### C3: Extractive QA Engine (`src/qa_engine.py`)
Multi-factor sentence scoring combining coverage, density, and length. Includes a configurable refusal threshold θ for unanswerable question detection.

### C4: Audit Logger (`src/audit.py`)
Structured JSON logging of every pipeline run. Fixed seed (s=42) ensures bitwise deterministic reproducibility.

## Evaluation

Download SQuAD v2.0 dev set:
```bash
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json -O data/dev-v2.0.json
```

Run full evaluation:
```bash
python -m src.evaluate --data data/dev-v2.0.json
```

Run ablation studies:
```bash
python -m src.evaluate --data data/dev-v2.0.json --ablation weights
python -m src.evaluate --data data/dev-v2.0.json --ablation threshold
python -m src.evaluate --data data/dev-v2.0.json --ablation chunksize
```

## Citation

```bibtex
@inproceedings{doifode2026sustainable,
  author    = {Doifode, Sayali Nitin},
  title     = {Sustainable Extractive Question Answering for Resource-Constrained 
               Personalized Document Assistants},
  booktitle = {Joint Proceedings of the ACM UMAP Workshops 2026 (LLM4Good)},
  year      = {2026},
  address   = {Gothenburg, Sweden}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
