"""Retrieval regression suite.

TWO TIERS
---------
  fast (default, every push)
      ~12 questions, deterministic retrieval metrics only. No LLM judge.
      Cost: retrieval only (Multi-Query + HyDE still call the utility model,
      so it is not free - roughly 2 utility calls per question).
      Runtime target: under 2 minutes.

  full (nightly / on-demand, marked `slow`)
      Whole golden set plus the ablation. Still no judge - RAGAS runs
      separately because judge variance makes it a poor build gate.

  Run fast:  pytest tests/test_retrieval_regression.py -m "not slow"
  Run full:  pytest tests/test_retrieval_regression.py

WHY RAGAS IS NOT A BUILD GATE
-----------------------------
An LLM judge has run-to-run spread. A threshold tight enough to catch a real
regression will also fire on judge noise, and a suite that cries wolf gets
ignored or disabled within two weeks. RAGAS belongs on a dashboard you look at,
not on a gate that blocks merges.

THRESHOLDS
----------
Every threshold below is DERIVED FROM A MEASURED BASELINE MINUS HEADROOM.
Fill them in from eval/results/retrieval_latest.json after your first run.
Do not set aspirational values - a gate you cannot pass today is a gate you
will delete tomorrow.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from eval import config as ev
from eval.retrieval_eval import _identity_rerank, _REAL_RERANK, rank_of_anchor
from app.core import config as app_config
from app.services import retrieval

# --- Thresholds ---------------------------------------------------------------
# TODO: replace the None values with (measured baseline - headroom) after the
# first real run. Tests fail loudly until you do, which is intentional.
#
# Suggested headroom: retrieval here is deterministic, so the only variance
# comes from Multi-Query/HyDE LLM output. Observed spread across 3 runs should
# set the headroom. Start with 0.10 absolute below baseline and tighten once
# you have seen the actual run-to-run movement.

BASELINE = json.loads(
    (ev.RESULTS_DIR / "retrieval_latest.json").read_text()
) if (ev.RESULTS_DIR / "retrieval_latest.json").exists() else None

MIN_HIT_RATE_AT_5 = None      # e.g. 0.72 if measured 0.82, headroom 0.10
MIN_HIT_RATE_AT_3 = None
MIN_MRR = None


def _load_golden(limit=None):
    if not ev.GOLDEN_SET_PATH.exists():
        pytest.skip("golden set not built")
    items = json.loads(ev.GOLDEN_SET_PATH.read_text())["items"]
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
    yield
    retrieval.rerank = _REAL_RERANK


def test_thresholds_are_configured():
    """Fails until thresholds come from a real measurement."""
    assert MIN_HIT_RATE_AT_5 is not None, (
        "Set thresholds from eval/results/retrieval_latest.json. "
        "A regression suite with no thresholds is decoration."
    )


# --- Fast tier ----------------------------------------------------------------

def test_fast_tier_hit_rate():
    items = _load_golden(ev.FAST_TIER_SIZE)
    metrics = _evaluate(items)
    print(f"\nfast tier ({len(items)} q): {metrics}")
    assert metrics["hit_rate@5"] >= MIN_HIT_RATE_AT_5, (
        f"hit_rate@5 {metrics['hit_rate@5']:.3f} below threshold "
        f"{MIN_HIT_RATE_AT_5}"
    )


def test_fast_tier_mrr():
    items = _load_golden(ev.FAST_TIER_SIZE)
    metrics = _evaluate(items)
    assert metrics["mrr"] >= MIN_MRR, f"MRR {metrics['mrr']:.3f} below {MIN_MRR}"


def test_config_has_not_drifted():
    """Metrics are only comparable if the configuration that produced them holds."""
    if not BASELINE:
        pytest.skip("no baseline recorded")
    recorded = BASELINE["config"]
    assert app_config.RETRIEVER_K == recorded["retriever_k"]
    assert app_config.RERANK_TOP_N == recorded["rerank_top_n"]
    assert app_config.CHUNK_SIZE == recorded["chunk_size"]
    assert app_config.EMBEDDING_MODEL == recorded["embedding_model"]


# --- Negative test ------------------------------------------------------------

@pytest.mark.negative
def test_suite_detects_degraded_retrieval():
    """THE test that proves the suite works.

    Disables re-ranking and asserts the fast tier would now FAIL. A regression
    suite that has never been seen to fail is not known to work.
    """
    items = _load_golden(ev.FAST_TIER_SIZE)
    retrieval.rerank = _identity_rerank
    degraded = _evaluate(items)
    retrieval.rerank = _REAL_RERANK
    healthy = _evaluate(items)

    print(f"\ndegraded: {degraded}\nhealthy:  {healthy}")
    assert healthy["hit_rate@5"] >= MIN_HIT_RATE_AT_5, "healthy arm should pass"
    assert degraded["hit_rate@5"] < MIN_HIT_RATE_AT_5, (
        "Degrading retrieval did NOT trip the threshold. Either the threshold "
        "is too loose, or re-ranking is contributing less than assumed. Both "
        "are findings worth writing down."
    )


# --- Full tier ----------------------------------------------------------------

@pytest.mark.slow
def test_full_golden_set():
    items = _load_golden()
    metrics = _evaluate(items)
    print(f"\nfull tier ({len(items)} q): {metrics}")
    assert metrics["hit_rate@5"] >= MIN_HIT_RATE_AT_5
    assert metrics["hit_rate@3"] >= MIN_HIT_RATE_AT_3
