# Variant 3 experiment results

**Variant 3: On-device QA model and prompt engineering.**
Local RAG pipeline with a small on-device LLM, evaluated on the first 100 answerable questions of SQuAD v2.0 dev.

Unanswerable questions are out of scope for this thesis and are filtered from all evaluations. The source SQuAD v2.0 file is unmodified. The filtering happens in memory at load time via the `--answerable-only` flag.

---

## 1. Evaluation setup

**Dataset.** SQuAD v2.0 dev — first 100 answerable questions in file order (deterministic).

**Machine.** Single CPU-only laptop, offline (no network calls except for the closed-baseline reference in Stage 6). Determinism enforced with temperature 0, seed 42, and a versioned prompt template (`variant3-2026-06-14`).

**Metric set.**

- **`answer_containment`** — primary quality metric. Fraction of predictions containing the gold answer.
- **`mean_latency_ms`, `p95_latency_ms`** — average and p95 response time.
- **`mean_context_tokens`, `mean_retrieved_chunks`** — describe how much evidence the LLM saw.
- **`generated` / `fallbacks`** — how many of the 100 items the LLM completed vs. how many fell through to the extractive fallback (Stage 1 JSONs only).

Metrics not reported: exact match (near zero by design — the citation format prevents literal string matches).

---

## 2. Configuration experiments

Each stage fixes all decisions from prior stages and varies exactly one knob.

### Stage 1 · Model selection

**Fixed:** `prompt_id = rag_cited_answer`, `top_k = 5`, chunker default (window 400 / overlap 80), `latency_target = 10000 ms`.
**Varied:** `--model`.

| Model         | Containment | Mean latency | p95 latency | Generated / 100 |
|---------------|-------------|--------------|-------------|-----------------|
| qwen3:0.6b    | 0.62        | 1,302 ms     | 2,871 ms    | 100             |
| **qwen3:1.7b**| **0.92**    | 4,229 ms     | 8,160 ms    | 98              |
| llama3.2:1b   | 0.54        | 3,569 ms     | 8,984 ms    | 97              |

**Decision: qwen3:1.7b.** It scored 30 points above qwen3:0.6b and 38 above llama3.2:1b. Its latency remained within the test budget, and it uses the Apache 2.0 licence.

### Stage 2 · Evidence count (K)

**Fixed:** `model = qwen3:1.7b`, `prompt_id = rag_cited_answer`, chunker reduced to window 100 / overlap 20 (so paragraphs produce multiple chunks and K meaningfully varies), `latency_target = 10000 ms`, `max_context_tokens = 4000`.
**Varied:** `--top-k`.

| K    | Containment | Mean latency | p95 latency | Avg chunks retrieved |
|------|-------------|--------------|-------------|----------------------|
| 1    | 0.90        | 4,072 ms     | 7,595 ms    | 1.00                 |
| **3**| **0.94**    | 4,195 ms     | 9,284 ms    | 1.81                 |
| 5    | 0.94        | 4,131 ms     | 7,444 ms    | 1.84                 |

**Decision: K = 3.** K = 5 gave the same containment, while K = 1 was four points lower. The average paragraph produced fewer than two retrieved chunks.

### Stage 3 · Prompt template

**Fixed:** `model = qwen3:1.7b`, `top_k = 3`, chunker 100 / 20, `latency_target = 10000 ms`, `max_context_tokens = 4000`, `self_correct = False`.
**Varied:** `--prompt-id`.

| Prompt               | Containment | Mean latency | p95 latency |
|----------------------|-------------|--------------|-------------|
| **rag_cited_answer** | **0.84**    | 9,353 ms     | 10,040 ms   |
| rag_strict           | 0.83        | 9,411 ms     | 10,030 ms   |

**Run conditions.** Both prompt runs were close to the 10-second limit and were recorded while the laptop was under load. A later repeat of `rag_cited_answer` reached 0.94 containment with 2,839 ms mean latency. The direct comparison therefore supports only a limited conclusion: `rag_strict` did not improve containment in this test.

**Decision: rag_cited_answer.** The strict prompt did not improve containment in this comparison, so the cited-answer prompt was retained.

### Stage 4 · Self-correction (second-pass refinement)

**Fixed:** `model = qwen3:1.7b`, `top_k = 3`, `prompt_id = rag_cited_answer`, chunker 100 / 20, `latency_target = 10000 ms`, `max_context_tokens = 4000`.
**Varied:** `--self-correct` off vs. on.

| Self-correct | Containment | Mean latency | p95 latency |
|--------------|-------------|--------------|-------------|
| **off**      | **0.84**    | 9,353 ms     | 10,040 ms   |
| on           | 0.87        | 9,664 ms     | 10,035 ms   |

**Interpretation.** Both runs were close to the 10-second limit. Self-correction added three containment points and 311 ms to mean latency in this test.

**Decision: self-correct off.** It requires a second model call and does not fit reliably within the final five-second budget.

### Stage 5 · Latency budget (fallback threshold)

**Fixed:** `model = qwen3:1.7b`, `top_k = 3`, `prompt_id = rag_cited_answer`, `self_correct = False`, chunker 100 / 20, `max_context_tokens = 4000`.
**Varied:** `--latency-target-ms`.

| Budget       | Containment | Mean latency | p95 latency | Regime          |
|--------------|-------------|--------------|-------------|-----------------|
| 1,500 ms     | 0.73        | 1,515 ms     | 1,533 ms    | All fallback    |
| 2,500 ms     | 0.73        | 2,517 ms     | 2,540 ms    | All fallback    |
| **5,000 ms** | **0.87**    | 4,215 ms     | 5,032 ms    | LLM engaged     |
| 10,000 ms    | 0.94        | 2,839 ms     | 5,445 ms    | Unconstrained   |

**Observed behaviour.** At 1.5 and 2.5 seconds, every answer used the extractive fallback and containment was 0.73. At five seconds, containment rose to 0.87. The ten-second run reached 0.94.

**Decision: 5,000 ms.** This kept p95 latency near five seconds and reached 0.87 containment.

### Stage 6 · Closed-LLM baseline (OpenAI GPT-4o-mini)

The closed model was used only as a reference. It received the SQuAD passage and question in a single request, with no retrieval or prompt tuning.

Predictions from GPT-4o-mini were saved to `results/closed_baseline_openai.csv` and attached to a local benchmark run via `--closed-baseline-csv`.

**Local system vs closed reference (`final_with_baseline.json`, 5-second budget):**

| System                                  | Containment | Token F1 | Exact match | Mean latency | p95 latency |
|-----------------------------------------|-------------|----------|-------------|--------------|-------------|
| Our local system (5 s budget, offline)  | 0.87        | —        | —           | 3,851 ms     | 5,060 ms    |
| GPT-4o-mini (closed API, single-shot)   | 0.98        | 0.76     | 0.52        | (API-side, not measured)                 |

**Exact match is not reported for the local system.** The `rag_cited_answer` prompt produces citation-formatted output (e.g. `"According to [C1], ..."`), which prevents literal string match against SQuAD gold spans. Containment is the intended primary metric and is directly comparable.

---

## 3. Final configuration

| Parameter                | Value                             |
|--------------------------|-----------------------------------|
| Model                    | `qwen3:1.7b` (Apache 2.0)         |
| Retrieval count (K)      | 3                                 |
| Chunker window / overlap | 100 / 20 tokens                   |
| Max context tokens       | 4,000                             |
| Prompt template          | `rag_cited_answer`                |
| Self-correction          | off                               |
| Latency budget           | 5,000 ms                          |
| Temperature / seed       | 0 / 42                            |
| Evaluation set           | SQuAD v2.0 dev, first 100 answerable |

---

## 4. Main result

| Level                        | Containment | Meaning                                                    |
|------------------------------|-------------|------------------------------------------------------------|
| Extractive fallback          | **0.73**    | Retrieval-only answer                                      |
| **Our local system (5 s)**   | **0.87**    | qwen3:1.7b + RAG at 5-second budget, offline CPU           |
| GPT-4o-mini reference        | **0.98**    | Closed API, single request with the passage                 |

**Gap analysis.**

- Difference between the fallback and closed baseline: 0.98 − 0.73 = 0.25
- Improvement from the local model: 0.87 − 0.73 = 0.14
- The local model covers 56% of that difference
- Its containment is 89% of the closed baseline (0.87 / 0.98)

---

## 5. Findings

1. **The 1.7B model gave the best local result.** It reached 0.87 containment on the 100-question test, compared with 0.98 for GPT-4o-mini.

2. **The extractive fallback remains useful.** It reached 0.73 containment without model generation and returned an answer when generation timed out.

3. **The extra prompt steps were not selected.** `rag_strict` did not improve the direct prompt comparison. Self-correction improved containment slightly but required another model call.

4. **The latency limit changes which path answers the question.** Short limits forced the fallback. A five-second limit allowed the local model to complete most requests.

5. **Model choice had the largest measured effect.** Qwen3-1.7B scored more than 30 containment points above the two smaller models under the same settings.

---

## 6. Conclusion

On the 100-question test, the local model raised answer containment from 0.73 for the extractive fallback to 0.87. GPT-4o-mini reached 0.98 on the same questions. The local configuration used:

- no network calls (fully offline)
- no API cost
- deterministic outputs (temperature 0, seed 42)
- bounded worst-case response time (p95 ≈ 5 s)
- a licence-permissive model (Qwen3, Apache 2.0)
- an extractive fallback when the model did not finish in time.

The experiments show that model choice, retrieval count, and the latency limit affected the result. The stricter prompt and self-correction were not used in the final five-second configuration because their measured gains did not justify the extra model work.

---

## Appendix: raw result files

All numbers above are drawn from JSON outputs in `results/`.

| Result reported          | JSON file                                                 |
|--------------------------|-----------------------------------------------------------|
| Model comparison         | `model_qwen3-0.6b.json`, `model_qwen3-1.7b.json`, `model_llama3.2-1b.json` |
| K sweep                  | `k_1.json`, `k_3.json`, `k_5.json`                        |
| Prompt comparison        | `prompt_cited.json`, `prompt_strict.json`                 |
| Self-correct comparison  | `prompt_cited.json`, `prompt_cited_selfcorrect.json`      |
| Latency budget sweep     | `budget_1500.json`, `budget_2500.json`, `budget_5000.json`, `budget_10000.json` |
| Closed baseline          | `closed_baseline_openai.csv` (predictions), `final_with_baseline.json` (combined evaluation) |

All local benchmark runs used the same 100 answerable questions in the same file order (SQuAD v2.0 dev). Reproduction commands are in the project README.
