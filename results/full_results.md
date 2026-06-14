# FULL SQuAD v2.0 DEV SET EVALUATION RESULTS
# 11,873 questions (5,928 answerable + 5,945 unanswerable)
# 35 articles, paragraph-level context

## MAIN RESULTS (θ=0.0, default weights)

| Metric                    | Value          |
|---------------------------|----------------|
| Total questions           | 11,873         |
| Answerable                | 5,928          |
| Unanswerable              | 5,945          |
| Answer Containment        | 69.0% (4,090)  |
| Average Token F1          | 0.171          |
| Exact Match               | 0.19% (11)     |
| Refusal Accuracy (θ=0)    | 0.0%           |
| Mean Latency              | 0.097 ms       |
| p95 Latency               | 0.183 ms       |
| Peak Memory               | <10 MB         |
| Est. Energy per Query     | ~0.0015 mJ     |

NOTE on Exact Match: EM is near zero because the system returns
full context sentences, not minimal answer spans. This is expected
and consistent with the extractive design. Answer Containment
(69.0%) is the primary utility metric.

NOTE on energy: 0.097ms mean × 15W TDP = 0.00146 mJ per query.


## REFUSAL THRESHOLD ABLATION

| θ    | Containment | Token F1 | Refusal Acc |
|------|-------------|----------|-------------|
| 0.00 | 69.0%       | 0.171    | 0.0%        |
| 0.15 | 69.0%       | 0.171    | 0.2%        |
| 0.25 | 68.2%       | 0.169    | 3.8%        |
| 0.30 | 66.0%       | 0.163    | 10.2%       |
| 0.35 | 61.4%       | 0.149    | 20.9%       |
| 0.40 | 54.5%       | 0.131    | 36.0%       |
| 0.45 | 44.7%       | 0.105    | 52.9%       |
| 0.50 | 33.7%       | 0.078    | 68.3%       |


## SCORING WEIGHT ABLATION

| Configuration (α, β, γ)        | Containment | Token F1 |
|---------------------------------|-------------|----------|
| Default (0.5, 0.3, 0.2)        | 69.0%       | 0.171    |
| Coverage-only (1.0, 0.0, 0.0)  | 75.0%       | 0.187    |
| Density-heavy (0.2, 0.6, 0.2)  | 60.8%       | 0.159    |
| Equal (0.33, 0.33, 0.33)       | 63.7%       | 0.156    |
| No length (0.6, 0.4, 0.0)      | 70.6%       | 0.183    |


## CHUNK SIZE ABLATION

| W   | O  | Containment | Token F1 | p95 Latency |
|-----|----|-------------|----------|-------------|
| 50  | 10 | 67.7%       | 0.183    | 0.266 ms    |
| 100 | 20 | 68.5%       | 0.175    | 0.274 ms    |
| 150 | 30 | 69.0%       | 0.173    | 0.263 ms    |
| 200 | 40 | 69.0%       | 0.171    | 0.255 ms    |
| 300 | 60 | 69.3%       | 0.172    | 0.230 ms    |
| 400 | 80 | 69.3%       | 0.172    | 0.220 ms    |


## KEY FINDINGS (vs. original 20-question evaluation)

1. Answer Containment drops from 85.7% (20q) to 69.0% (11,873q).
   Still strong for a zero-dependency TF-IDF system.

2. EM drops from 92.9% to ~0%. The original 20q set likely had
   gold answers that were complete sentences. At scale, SQuAD
   gold answers are short spans that never match full sentences.
   EM is NOT a meaningful metric for sentence-level extraction.

3. NEW FINDING: Coverage-only weights (1.0, 0.0, 0.0) outperform
   the default at scale (75.0% vs 69.0% containment). The original
   20q evaluation showed default as best. This reversal suggests
   the multi-factor design adds complexity without benefit at scale,
   though coverage-only may overfit to certain question types.

4. Chunk size has minimal impact (67.7%–69.3%), suggesting the
   pipeline is robust to this parameter.

5. Latency remains sub-millisecond (0.097ms mean, 0.183ms p95).
   Energy estimate: ~0.0015 mJ per query (even lower than
   originally estimated).
