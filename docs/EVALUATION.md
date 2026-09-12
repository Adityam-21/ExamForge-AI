# Evaluation

How ExamForge's retrieval quality is measured, what the numbers are, and what
they do not cover. Every figure here comes from a committed artifact in
`eval/results/` and can be reproduced with the commands at the bottom.

Measured 10-11 September 2026.

---

## What is measured, and what is not

| Question | Method | Judge | Status |
|---|---|---|---|
| Does retrieval find the right passage? | hit-rate@k, MRR vs. labelled anchor chunks | None - deterministic | Measured |
| Does re-ranking help? | A/B ablation at matched depth | None - deterministic | Measured |
| How fast is a full request? | Timed end-to-end pipeline runs | None | Measured |
| What does a request cost? | LangSmith token accounting | None | Measured |
| Does the answer stick to its sources? | RAGAS `faithfulness` | LLM | **Not run** - see Limitations |
| Is the answer factually correct? | Needs reference answers | - | Not measured |

---

## The golden set

`eval/golden_set.json` - 39 questions.

**Corpus.** NCERT Class 10 Science, 11 chapter PDFs. 10 ingested successfully
(883 chunks); `jesc104.pdf` was skipped - pypdf raises `PdfReadError` on a
malformed content stream. The skip is recorded in
`eval/results/corpus_manifest.json` alongside per-document content hashes, so
the corpus is identifiable without redistributing the PDFs.

One subject was chosen deliberately: adjacent chapters share vocabulary, so
retrieval has to distinguish genuinely similar passages rather than just pick
the right document.

**Construction - chunk-anchored generation with human verification.**

1. 60 chunks sampled at random from the 779 above 300 characters (seed `20260910`).
2. An LLM wrote one exam-style question per chunk. Because the question was
   written *from* that chunk, the chunk is by construction a relevant passage
   for it. That is the ground-truth label - it is a structural fact about how
   the question was produced, not a model's opinion about relevance.
3. **Every candidate was reviewed by hand**, with the source passage visible,
   and accepted, edited or rejected. 51 reviewed, 40 accepted (78%).
4. One accepted question (`q003`) was later removed: it shared an anchor page
   with `q045` (jesc109 p13, same activity). Since each question carries exactly
   one anchor label, a correct retrieval of the *other* glass-slab chunk would
   have scored as a miss. Final set: **39**.

Rejected for: referencing something unseen ("the 5-ohm resistor", "the line"),
being a textbook calculation answerable without retrieval, or being a near
duplicate of an already-accepted question.

**Labels.** One anchor chunk per question, keyed `doc_id::page::chunk_index`.
Page is part of the key because `chunk_index` restarts at 0 on every page
(`ingestion.py` enumerates inside the per-page split loop) - without it, chunks
from different pages collide.

**Reference answers.** None. This is what rules out `context_recall` and
`answer_correctness`.

---

## Results

### Retrieval - re-ranking ablation

`python -m eval.retrieval_eval` - raw: `eval/results/retrieval_latest.json`

Two arms, identical except that one replaces `rerank()` with an identity
function truncating to the same depth (`RERANK_TOP_N=5`). Depth is matched
deliberately: comparing top-5 reranked against top-20 fused would measure
truncation, not re-ranking. The swap happens at runtime in the eval harness -
**no source file is modified**, so the reranked arm exercises the exact
production code path.

Arms are **interleaved per question**, not run in sequence, so network drift and
rate limiting affect both equally.

| Metric | Fusion only | + FlashRank | Change |
|---|---|---|---|
| hit-rate@1 | **0.641** | 0.231 | -64.0% |
| hit-rate@3 | **0.872** | 0.513 | -41.2% |
| hit-rate@5 | **0.949** | 0.692 | -27.0% |
| MRR | **0.766** | 0.399 | -0.367 |

n=39. Comparability guard passed: 0 questions with a candidate-pool mismatch,
0 retrieval errors, mean candidates 11.5 (fusion) vs 11.7 (reranked).

**FlashRank degrades retrieval on this corpus.** Nine questions that fusion
returned at rank 1 were pushed out of the top 5 entirely. The result replicated
across three independent runs (two full, one on a 12-question subset) and again
on GitHub Actions.

Two plausible mechanisms, neither yet isolated:

- **The candidate pool is small.** `EnsembleRetriever` already applies reciprocal
  rank fusion over dense + BM25, so first-stage order is good. Re-ranking ~11
  well-ordered candidates down to 5 gives a cross-encoder little room to add
  value and plenty to do harm.
- **The default FlashRank model is small.** `ms-marco-MultiBERT-L-12`, trained on
  clean web passages. These chunks are PDF extractions containing artifacts
  (`/square6` bullets, broken ligatures, `RESPIR A AAAATION`). A small
  cross-encoder reading noisy text may score it poorly where BGE embeddings
  degrade more gracefully.

**Action taken.** `RERANK_ENABLED` defaults to `false`. The reranker is kept
behind the flag so the ablation stays reproducible and the regression suite can
use it as a known-bad arm.

### Latency

`python -m eval.latency_cost --n 30` - raw: `eval/results/latency_latest.json`

| min | p50 | p90 | p95 | p99 | mean | stdev |
|---|---|---|---|---|---|---|
| 4.32s | 6.92s | 11.35s | **11.87s** | 13.76s | 7.69s | 2.62s |

- n=29 warm samples; the first request (8,695 ms) is discarded as cold and
  reported separately.
- Percentiles by nearest rank. At n=29, p95 is the second-slowest sample - a
  thin estimate.
- Measured: full `run_pipeline` - condensation, 4 retrieval strategies, fusion,
  generation streamed to completion.
- **Not** measured: HTTP overhead, SSE transport, frontend render, user network.
- Local machine, Groq free tier. **Latency drifted upward during the run**
  (early ~5s, later ~11s) as rate-limit backoff engaged. This number therefore
  includes throttling and is not a clean server-side measurement.

### Cost

From a LangSmith trace of one end-to-end request
(`docs/artifacts/langsmith-trace-tree.png`):

```
LangGraph              10.16s   2.21K tokens   $0.0007
├─ condense             0.00s                  (no history on this turn)
└─ retrieve             7.79s     791 tokens
   ├─ EnsembleRetriever 0.42s
   │  ├─ BM25Retriever        0.02s
   │  └─ VectorStoreRetriever 0.40s
   ├─ MultiQueryRetriever 1.73s   431 tokens
   └─ HyDE chain          ~1.6s
```

**~$0.0007 (~Rs 0.06) per query.** Single trace, so indicative rather than a
distribution.

**Retrieval is 77% of end-to-end latency** (7.79s of 10.16s) - not generation.
Multi-Query alone costs 1.73s for one LLM call.

### Configuration these numbers were measured under

| | |
|---|---|
| Embedding | `BAAI/bge-small-en-v1.5` |
| Generation | `openai/gpt-oss-120b` (Groq) |
| Utility (Multi-Query, HyDE) | `openai/gpt-oss-20b` (Groq) |
| Chunking | 512 chars, 50 overlap, recursive |
| `RETRIEVER_K` | 5 |
| `RERANK_TOP_N` | 5 |
| `RERANK_ENABLED` | false |
| `MIN_RELEVANCE_SCORE` | 0.05 |

Every LangSmith run is tagged with these via `_traced_pipeline` in
`app/api/routes.py`. Untagged runs cannot be compared across configurations.

---

## Regression suite

`tests/test_retrieval_regression.py`, two tiers.

**Fast** - 12 questions, deterministic metrics only, every push.
**Full** - all 39, nightly and on demand.
**Negative** - deliberately degrades retrieval and asserts the gate trips.

### Thresholds

| Gate | Threshold | Measured baseline | Headroom |
|---|---|---|---|
| hit-rate@5 | 0.83 | 0.949 | 0.12 |
| hit-rate@3 | 0.74 | 0.872 | 0.13 |
| MRR | 0.65 | 0.766 | 0.12 |

Headroom is wide on purpose. The fast tier runs 12 questions, so **one question
moves hit-rate by 8.3 points**. Multi-Query and HyDE are LLM calls whose output
varies run to run - CI and local produced identical *healthy* numbers but
differed on the degraded arm (hit-rate@3 0.500 vs 0.417). A gate tighter than
one-question-plus-noise would fire on normal variation, and a suite that cries
wolf gets disabled within a fortnight.

### Why RAGAS is not a build gate

An LLM judge has run-to-run spread. A threshold tight enough to catch a real
regression also fires on judge noise. Generation-quality metrics belong on a
dashboard you look at, not on a gate that blocks merges.

### Negative test

Re-enables FlashRank - a measured, real degradation rather than a synthetic one -
and asserts the fast-tier gate now fails.

```
degraded (FlashRank on): hit_rate@3 0.500  hit_rate@5 0.583  mrr 0.336
healthy  (production):   hit_rate@3 0.833  hit_rate@5 0.917  mrr 0.8125
PASSED
```

A regression suite that has never been seen to fail is not known to work.

### Reproducibility

The fast tier produces **identical results locally and in GitHub Actions**:

```
fast tier (n=12): hit_rate@3 0.8333  hit_rate@5 0.9167  mrr 0.8125
```

Local: Windows, Python 3.11.9, `sentence-transformers` 6.0.1.
CI: Ubuntu, Python 3.11.16, `sentence-transformers` 3.0.1 (pinned in
`requirements.txt`).

Same numbers to four decimal places despite a major version gap in
`sentence-transformers`, which also rules that out as a source of drift.

---

## Faults this work uncovered

**1. `UTILITY_MODEL` had been decommissioned.** `llama-3.1-8b-instant` was
deprecated by Groq in June 2026 and later removed. Multi-Query and HyDE both run
inside `_safe()`, which catches the exception, logs a warning and returns `[]`.
Production had therefore been running on **two of four retrieval strategies for
roughly three months**, with no error surfaced to users and no alert. The
fault-tolerant design hid a real degradation. A trace view would have shown two
retrieval spans returning zero documents on every request.

**2. The abstention gate was coupled to the reranker.** `relevance_of()` read
only `relevance_score`, written by FlashRank. Disabling the reranker left it
reading 0.0, so `best_score < MIN_RELEVANCE_SCORE` abstained on **every**
question - the app answered nothing while retrieval metrics stayed green.
Fixed by scoring chunks with cosine similarity against the same BGE embeddings
when the reranker is off (`_score_by_similarity`).

The regression suite did not catch this, because it measures retrieval ranks and
never asserts that the pipeline produces an answer. A smoke test asserting
`len(citations) > 0` is in the backlog.

---

## Limitations

Read these before quoting any number above.

**1. Single-label recall is a lower bound.** Each question has one labelled
chunk. Another chunk may answer it just as well; retrieving that one scores as a
miss. The reported figures understate true retrieval quality.

**2. Vocabulary bias inflates the scores.** Questions were generated from the
chunks they are labelled against, so they share vocabulary with them. This
flatters both lexical and dense retrieval relative to how a student would
actually phrase a question. Manual review rejected the worst offenders but
cannot eliminate the effect. Fusion-only hit-rate@5 of 0.949 should be read as
"near-ceiling on a set biased toward easy retrieval", not "94.9% of real student
questions succeed".

**3. One corpus, one subject.** Results are specific to these 10 NCERT chapters.
The FlashRank finding in particular may not generalise to cleaner text or a
larger candidate pool.

**4. Small n.** At 39 questions, one question is worth 2.6 percentage points of
hit-rate. At 12 (fast tier), 8.3 points. Differences smaller than that are noise.

**5. The abstention gate is weakened in similarity mode.** FlashRank scores and
cosine similarities are on different scales, but `MIN_RELEVANCE_SCORE` is still
0.05 - a value tuned for FlashRank. BGE cosine rarely falls below 0.05 on this
corpus, so the pre-generation abstention no longer discriminates: "What is the
capital of Brazil?" returns 5 citations instead of abstaining. Downstream
`is_grounded` still returns `False` and the UI shows a low-confidence warning,
so it degrades rather than fails. **Calibrating this threshold requires a
negative question set that does not yet exist** and is in the backlog. It has
not been guessed at.

**6. RAGAS was not run.** `faithfulness` and `answer_relevancy` were scoped and
`eval/ragas_eval.py` is written against the installed RAGAS 0.1.19 API, but it
cannot execute on the development machine: a Windows Application Control policy
blocks DLLs in the `pyarrow` dependency chain that `datasets` requires, and
`datasets` is in turn required by `sentence-transformers`. Installing one breaks
the other. **No generation-side quality metric has been measured.** Hallucination
rate is unknown.

**7. Latency was measured under rate limiting.** See above - the free-tier
throttle inflates the tail. A clean measurement needs a paid tier or a slower
sampling rate.

---

## Reproducing

```bash
pip install -r requirements.txt
pip install pytest
export GROQ_API_KEY=...

python -m eval.setup_corpus        # ingest eval/corpus/ into eval-corpus-v1
python -m eval.retrieval_eval      # ablation, both arms, interleaved
python -m eval.latency_cost --n 30 # p50/p95 with method recorded

pytest tests/test_retrieval_regression.py -m "not slow and not negative"
pytest tests/test_retrieval_regression.py -m negative
```

`.github/workflows/eval.yml` runs the fast tier and the negative test on every
push, and the full tier nightly. It pins `RERANK_ENABLED=false` and
`UTILITY_MODEL` explicitly - CI running a different configuration than the
baseline would make the comparison meaningless.

The golden set, the corpus, and all results in `eval/results/` are committed. A
metric whose evidence is not in the repository is just a number.

---

## Backlog

Deliberately not acted on during measurement, to keep instrumentation and
behaviour change separable.

1. **Calibrate `MIN_RELEVANCE_SCORE` for cosine similarity** - build ~20 off-corpus
   questions, find the threshold that separates them from real ones. (Limitation 5.)
2. **Smoke test**: one question, assert `len(citations) > 0`. Would have caught
   fault 2 immediately.
3. **Investigate the FlashRank result** - larger candidate pool (`RETRIEVER_K`
   5 -> 20), a stronger FlashRank model, or PDF extraction cleanup. Re-measure
   before and after.
4. **Question whether Multi-Query and HyDE earn their cost** - 1.73s and ~1.6s
   respectively, on a pipeline where retrieval is already 77% of latency, and
   which ran without them for three months. Ablate each.
5. **Startup model-availability check** - fail loudly when a configured model has
   been decommissioned, instead of degrading silently. (Fault 1.)
6. **Handle malformed PDFs** in `ingest_pdf` - `jesc104.pdf` 500s the upload
   endpoint today.
7. **Reference answers** for 20-30 questions to unlock `context_recall`.
