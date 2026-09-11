"""Retrieval regression suite.

TWO TIERS
---------
  fast (every push)
      12 questions, deterministic retrieval metrics only. No LLM judge.
      Multi-Query and HyDE still call the utility model, so it is not free.

  full (nightly / on-demand, marked `slow`)
      The whole golden set.

  Run fast:      python -m pytest tests/test_retrieval_regression.py -m "not slow and not negative"
  Run negative:  python -m pytest tests/test_retrieval_regression.py -m negative
  Run full:      python -m pytest tests/test_retrieval_regression.py -m "not negative"

WHY RAGAS IS NOT A BUILD GATE
-----------------------------
An LLM judge has run-to-run spread. A threshold tight enough to catch a real
regression also fires on judge noise, and a suite that cries wolf gets disabled
within a fortnight. RAGAS belongs on a dashboard, not on a merge gate.

THRESHOLDS
----------
Measured baselines (fusion arm, eval/results/retrieval_latest.json, n=39):
    hit_rate@5 = 0.949    hit_rate@3 = 0.872    MRR = 0.766

Thresholds sit ~0.12 absolute below those. That headroom is deliberately wide:
  - the fast tier runs 12 questions, so ONE question moves hit-rate by 8.3 pts
  - Multi-Query and HyDE are LLM calls whose output varies between runs
A tighter gate would fire on normal variation rather than on regressions.
"""

from __future__ import annotations

import json

import pytest

from eval import config as ev
from eval.retrieval_eval import _REAL_RERANK, rank_of_anchor
from app.core import config as app_config
from app.services import retrieval

# --- Thresholds --------------------------------------------------------------

MIN_HIT_RATE_AT_5 = 0.83
MIN_HIT_RATE_AT_3 = 0.74
MIN_MRR = 0.65

_BASELINE_PATH = ev.RESULTS_DIR / "retrieval_latest.json"
BASELINE = (
    json.loads(_BASELINE_PATH.read_text(encoding="utf-8"))
    if _BASELINE_PATH.exists()
    else None
)

def _load_golden(limit=None):
    if not ev.GOLDEN_SET_PATH.exists():
        pytest.skip("golden set not built")
    items = json.loads(ev.GOLDEN_SET_PATH.read_text(encoding="utf-8"))["items"]
    return items[:limit] if limit else items


def _evaluate(items):
    ranks = []
    for item in items:
        results = retrieval.retrieve(ev.EVAL_SESSION_ID, item["question"])
        ranks.append(rank_of_anchor(results, set(item["relevant_chunk_ids"])))
    n = len(ranks)
    return {
        "hit_rate@3": sum(1 for r in ranks if r and r <= 3) / n,
        "hit_rate@5": sum(1 for r in ranks if r and r <= 5) / n,
        "mrr": sum((1 / r) if r else 0.0 for r in ranks) / n,
    }


@pytest.fixture(autouse=True)
def _restore_reranker():
    """The production reranker is always restored, even if a test fails."""
    yield
    retrieval.rerank = _REAL_RERANK


# --- Fast tier ---------------------------------------------------------------


def test_fast_tier_retrieval_quality():
    items = _load_golden(ev.FAST_TIER_SIZE)
    metrics = _evaluate(items)
    print(f"\nfast tier (n={len(items)}): {metrics}")
    assert (
        metrics["hit_rate@5"] >= MIN_HIT_RATE_AT_5
    ), f"hit_rate@5 {metrics['hit_rate@5']:.3f} below threshold {MIN_HIT_RATE_AT_5}"
    assert (
        metrics["mrr"] >= MIN_MRR
    ), f"MRR {metrics['mrr']:.3f} below threshold {MIN_MRR}"


def test_config_has_not_drifted():
    """Metrics are only comparable if the config that produced them still holds."""
    if not BASELINE:
        pytest.skip("no baseline recorded")
    recorded = BASELINE["config"]
    assert app_config.RETRIEVER_K == recorded["retriever_k"]
    assert app_config.RERANK_TOP_N == recorded["rerank_top_n"]
    assert app_config.CHUNK_SIZE == recorded["chunk_size"]
    assert app_config.EMBEDDING_MODEL == recorded["embedding_model"]


# --- Negative test -----------------------------------------------------------


@pytest.mark.negative
def test_suite_detects_degraded_retrieval():
    """Proves the suite can fail. A gate never seen to fail is not known to work.

    The degradation is re-enabling FlashRank, which measurement showed cuts
    retrieval quality roughly in half on this corpus.
    """
    items = _load_golden(ev.FAST_TIER_SIZE)

    original = app_config.RERANK_ENABLED
    try:
        app_config.RERANK_ENABLED = True
        degraded = _evaluate(items)
    finally:
        app_config.RERANK_ENABLED = original
    healthy = _evaluate(items)

    print(f"\ndegraded (FlashRank on): {degraded}")
    print(f"healthy  (production):    {healthy}")

    assert healthy["hit_rate@5"] >= MIN_HIT_RATE_AT_5
    assert degraded["hit_rate@5"] < MIN_HIT_RATE_AT_5

    retrieval.rerank = _REAL_RERANK


# --- Full tier ---------------------------------------------------------------


@pytest.mark.slow
def test_full_golden_set():
    items = _load_golden()
    metrics = _evaluate(items)
    print(f"\nfull tier (n={len(items)}): {metrics}")
    assert metrics["hit_rate@5"] >= MIN_HIT_RATE_AT_5
    assert metrics["hit_rate@3"] >= MIN_HIT_RATE_AT_3
