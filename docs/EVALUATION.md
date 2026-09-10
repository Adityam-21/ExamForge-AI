# Evaluation

> **Template.** Every `___` is a blank to fill from a real run. Do not fill any
> of them from memory, intuition, or a previous version of this document.

## What is measured, and what is not

| Question | Method | Judge |
|---|---|---|
| Does retrieval find the right passage? | hit-rate@k, MRR vs. labelled anchor chunks | None — deterministic |
| Does re-ranking help? | A/B ablation, identical depth | None — deterministic |
| Does the answer stick to its sources? | RAGAS `faithfulness` | LLM (`___`) |
| Does the answer address the question? | RAGAS `answer_relevancy` | LLM (`___`) |
| How fast? | Timed end-to-end pipeline runs | None |
| Is the answer *correct*? | **Not measured** — needs reference answers | — |

## The golden set

- **Size:** `___` questions
- **Corpus:** `___` PDFs, `___` chunks, session `eval-corpus-v1`
- **Construction:** chunk-anchored generation with human verification.
  Chunks were sampled at random (seed `20260910`); an LLM wrote one question per
  chunk; **every question was then reviewed by hand** and accepted, edited or
  rejected. Acceptance rate: `___`%.
- **Labels:** one anchor chunk per question (`doc_id::chunk_index`).
- **Reference answers:** none.

### Known limitations — read these before quoting any number

1. **Single-label recall is a lower bound.** Other chunks may also be relevant;
   they aren't labelled, so a "miss" sometimes means "found a different good
   passage". The number understates true retrieval quality.
2. **Vocabulary bias.** Questions were written from the chunks, so they share
   vocabulary with them. This *flatters* lexical and dense retrieval relative to
   how a real student would phrase a question. Manual review rejected the worst
   offenders but cannot eliminate the effect.
3. **One corpus.** Results are specific to these `___` documents. They do not
   generalise to arbitrary study material.
4. **Judge variance.** RAGAS scores move between runs. Quote the range, never a
   single decimal.
5. **Small n.** At `___` questions, one question is worth `___` percentage points
   of hit-rate. Differences smaller than that are noise.

## Results

### Retrieval (deterministic)

Run: `python -m eval.retrieval_eval` · Raw: `eval/results/retrieval_latest.json`

| Metric | Fusion only | + FlashRank rerank | Δ |
|---|---|---|---|
| hit-rate@1 | `___` | `___` | `___` |
| hit-rate@3 | `___` | `___` | `___` |
| hit-rate@5 | `___` | `___` | `___` |
| MRR | `___` | `___` | `___` |

Both arms use identical retrieval depth (`RERANK_TOP_N=___`), so the delta
measures re-ranking, not truncation. The ablation is performed by substituting
the reranker at runtime in the eval harness; **no production code is modified.**

### Generation quality (LLM judge)

Run: `python -m eval.ragas_eval` · Raw: `eval/results/ragas_latest.json`

- Judge model: `___`, temperature 0
- Runs: `___` · Questions scored: `___` · Abstentions excluded: `___`

| Metric | Mean | Range | Spread |
|---|---|---|---|
| faithfulness | `___` | `___`–`___` | `___` |
| answer_relevancy | `___` | `___`–`___` | `___` |

### Latency and cost

Run: `python -m eval.latency_cost` · Raw: `eval/results/latency_latest.json`

- Sample: `___` warm requests, first discarded as cold (cold start `___`ms)
- Measured: full pipeline — condense, 4 retrieval strategies, rerank, generation
- **Not** measured: HTTP, SSE transport, frontend render, user network

| p50 | p90 | p95 | p99 |
|---|---|---|---|
| `___`s | `___`s | `___`s | `___`s |

Cost per query: `___` input + `___` output tokens → ₹`___`
(token counts from LangSmith, prices as of `___`)

## Thresholds

| Gate | Threshold | Measured baseline | Headroom | Why |
|---|---|---|---|---|
| fast hit-rate@5 | `___` | `___` | `___` | `___` |
| fast MRR | `___` | `___` | `___` | `___` |

Thresholds are baseline minus observed run-to-run spread. They are not
aspirational targets.

**Negative test.** `pytest -m negative` disables re-ranking and asserts the
fast tier fails, then restores it and asserts it passes. Result: `___`

## Reproducing

```bash
pip install -r requirements.txt
cp your-pdfs/*.pdf eval/corpus/
export GROQ_API_KEY=...
python -m eval.setup_corpus
python -m eval.retrieval_eval
python -m eval.ragas_eval
python -m eval.latency_cost
pytest tests/test_retrieval_regression.py
```

Golden set and results are committed. Corpus PDFs are not (`___` — state whether
for size or licensing).
