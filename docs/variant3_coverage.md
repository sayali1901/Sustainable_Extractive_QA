# Variant 3 Coverage

## Implemented

- Offline extractive QA fallback on CPU with no external dependencies.
- Optional local LLM generation through an already-installed Ollama runtime.
- Prompt pack:
  - `rag_cited_answer`
  - `rag_strict`
  - `short_summary`
  - `self_correct`
  - `graph_answer`
- Latency discipline:
  - Per-request target in milliseconds.
  - Generator timeout derived from remaining budget.
  - Retrieval-only fallback when no model is available or generation exceeds budget.
- Determinism:
  - Fixed seed.
  - Temperature set to `0.0` for Ollama generation.
  - Audit metadata for prompt, context size, token estimates, backend, model, and fallback reason.
- Multi-document support:
  - Multiple documents indexed into a shared retrieval space.
  - Citation labels preserve source document names.
- Graph prompts:
  - Deterministic keyword-to-chunk graph hints for retrieved context.
- Benchmarking:
  - `src.benchmark_variant3` compares prompt/model settings on SQuAD v2.0.
  - Reports accuracy, p95 latency, token counts, generated/fallback counts.
  - Closed-LLM baseline support uses saved predictions only; it does not tune closed models.
- UI:
  - Standard-library web UI.
  - Pasted document support or default project documents.
  - Answer, summary, citations, latency, token, prompt, and fallback display.

## Still Research/Environment Dependent

- Actual Qwen/Llama quality comparisons require local model files to be installed.
- Memory and energy should be measured on the target hardware profile, not inferred.
- Fine-tuning is not part of the lightweight runtime. Treat it as an offline preparation step and benchmark the resulting local model through the same harness.
- Closed-LLM baselines require externally generated prediction files. The runtime intentionally does not call closed APIs.

## Recommended Resource Tiers

- Tiny CPU tier: `qwen3:0.6b` through Ollama, strict latency target, short context.
- Small laptop tier: `qwen3:1.7b` or `qwen2.5:1.5b`, cited answer prompt, self-correction only when the target allows it.
- Medium laptop tier: `llama3.2:3b` or equivalent, graph prompt for multi-document tasks, larger context window.

Qwen models are preferred for this project when available because the listed
small Qwen candidates use Apache-2.0 licensing. Llama candidates are useful
baselines but have a more restrictive community licence.
