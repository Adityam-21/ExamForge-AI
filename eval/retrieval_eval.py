"""Deterministic retrieval evaluation with a re-ranking ablation.

This produces the honest replacement for the hit-rate claim. No LLM judge is
involved in scoring: a chunk either appears in the top-k or it does not.

EXPERIMENTAL DESIGN
-------------------
Two arms, identical except for one thing:

  arm "fusion"   - rerank() replaced by an identity function truncating to the
                   same depth. Ranking is first-stage fusion order.
  arm "reranked" - production path, FlashRank cross-encoder reordering.

The swap happens at runtime inside this script. NO SOURCE FILE IS MODIFIED, so
the "reranked" arm exercises the exact code path that runs in production.

INTERLEAVED, NOT SEQUENTIAL. Both arms run back-to-back for each question
before moving to the next. Running arm A fully and then arm B lets network
conditions, rate limits and cache state drift between them, and any drift gets
falsely attributed to re-ranking. Interleaving spreads that noise evenly.

CANDIDATE-COUNT GUARD. Multi-Query and HyDE run inside `_safe()` in
retrieval.py, which swallows exceptions and returns []. A rate limit or dropped
connection therefore shrinks the candidate pool *silently* - the run completes
and looks fine. This script records how many candidates each arm actually saw
and refuses to report an ablation if the arms were not comparable.

METRICS
-------
  hit_rate@k  - fraction of questions with the anchor chunk in the top k
  mrr         - mean reciprocal rank of the anchor chunk (0 if absent)

Each question has exactly one labelled chunk, so these are a LOWER BOUND on
true retrieval quality: an unlabelled but genuinely relevant chunk scores as a
miss.

Usage:
    python -m eval.retrieval_eval
    python -m eval.retrieval_eval --limit 12 --delay 3
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone

from eval import config as ev
from app.core import config as app_config
from app.services import retrieval

# Handle on the real reranker, captured before anything patches it.
_REAL_RERANK = retrieval.rerank

# Candidate counts observed per arm, filled by the recording wrappers below.
_candidates_seen: dict[str, list[int]] = {"fusion": [], "reranked": []}


def _make_identity_rerank():
    """First-stage fusion order, truncated to the same depth as the reranker.

    Depth is matched deliberately: comparing top-5 reranked against top-20
    fused would measure truncation, not re-ranking.
    """

    def _identity(question, chunks):
        _candidates_seen["fusion"].append(len(chunks))
        return chunks[: app_config.RERANK_TOP_N]

    return _identity


def _make_recording_rerank():
    """Production reranker, with the candidate count recorded on the way in."""

    def _recording(question, chunks):
        _candidates_seen["reranked"].append(len(chunks))
        return _REAL_RERANK(question, chunks)

    return _recording


def chunk_key(metadata: dict) -> str:
    """Stable identity for one chunk.

    `chunk_index` restarts at 0 on every page (ingestion.py enumerates within
    the per-page split loop), so page must be part of the key or chunks from
    different pages collide.
    """
    return (
        f"{metadata.get('doc_id')}"
        f"::{metadata.get('page')}"
        f"::{metadata.get('chunk_index')}"
    )


def rank_of_anchor(results, relevant_ids: set[str]) -> int | None:
    for position, chunk in enumerate(results, start=1):
        if chunk_key(chunk.metadata or {}) in relevant_ids:
            return position
    return None


def _run_one(arm: str, question: str, relevant: set[str]):
    """One retrieval for one arm. Returns (rank, seconds, n_results, error)."""
    retrieval.rerank = (
        _make_identity_rerank() if arm == "fusion" else _make_recording_rerank()
    )
    started = time.perf_counter()
    try:
        results = retrieval.retrieve(ev.EVAL_SESSION_ID, question)
    except Exception as exc:  # noqa: BLE001
        return None, time.perf_counter() - started, 0, str(exc)
    elapsed = time.perf_counter() - started
    return rank_of_anchor(results, relevant), elapsed, len(results), None


def _summarise(ranks: list[int | None], latencies: list[float]) -> dict:
    n = len(ranks)
    metrics = {
        f"hit_rate@{k}": round(sum(1 for r in ranks if r and r <= k) / n, 4)
        for k in ev.K_VALUES
    }
    metrics["mrr"] = round(sum((1 / r) if r else 0.0 for r in ranks) / n, 4)
    metrics["n_questions"] = n
    metrics["retrieval_p50_seconds"] = (
        round(statistics.median(latencies), 3) if latencies else None
    )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds to pause between questions. Groq free tier is 8000 TPM; "
        "each question fires ~2 utility calls per arm. Raise if you see 429s.",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if not ev.GOLDEN_SET_PATH.exists():
        print("No golden set. Run build_golden_set then review_golden_set.")
        return 1

    golden = json.loads(ev.GOLDEN_SET_PATH.read_text(encoding="utf-8"))
    items = golden["items"]
    if args.limit:
        items = items[: args.limit]

    print(f"Golden set: {len(items)} questions")
    print(
        f"Config: k={app_config.RETRIEVER_K} "
        f"rerank_top_n={app_config.RERANK_TOP_N} "
        f"chunk={app_config.CHUNK_SIZE}/{app_config.CHUNK_OVERLAP}"
    )
    print(f"Interleaved arms, {args.delay}s between questions\n")
    print(f"{'':>5} {'id':<8} {'fusion':>8} {'reranked':>10}  {'cand':>10}")
    print("-" * 50)

    per_arm = {
        "fusion": {"ranks": [], "latencies": [], "errors": 0},
        "reranked": {"ranks": [], "latencies": [], "errors": 0},
    }
    per_question = []

    for i, item in enumerate(items, start=1):
        relevant = set(item["relevant_chunk_ids"])
        row = {"id": item["id"]}

        for arm in ("fusion", "reranked"):
            rank, elapsed, n_results, error = _run_one(arm, item["question"], relevant)
            per_arm[arm]["ranks"].append(rank)
            if error:
                per_arm[arm]["errors"] += 1
            else:
                per_arm[arm]["latencies"].append(elapsed)
            row[arm] = {
                "rank": rank,
                "seconds": round(elapsed, 2),
                "n_results": n_results,
                "error": error,
            }

        f_cand = _candidates_seen["fusion"][-1] if _candidates_seen["fusion"] else 0
        r_cand = _candidates_seen["reranked"][-1] if _candidates_seen["reranked"] else 0
        row["candidates"] = {"fusion": f_cand, "reranked": r_cand}
        per_question.append(row)

        f_mark = f"@{row['fusion']['rank']}" if row["fusion"]["rank"] else "MISS"
        r_mark = f"@{row['reranked']['rank']}" if row["reranked"]["rank"] else "MISS"
        flag = "  <-- POOL MISMATCH" if abs(f_cand - r_cand) > 3 else ""
        print(
            f"{i:>5} {item['id']:<8} {f_mark:>8} {r_mark:>10}  "
            f"{f_cand:>4}/{r_cand:<4}{flag}"
        )

        retrieval.rerank = _REAL_RERANK
        if args.delay and i < len(items):
            time.sleep(args.delay)

    retrieval.rerank = _REAL_RERANK

    # --- Comparability guard -------------------------------------------------
    # If the two arms saw materially different candidate pools, the ablation is
    # confounded and the delta means nothing. Report it, do not hide it.
    f_counts = _candidates_seen["fusion"]
    r_counts = _candidates_seen["reranked"]
    mismatches = sum(1 for f, r in zip(f_counts, r_counts) if abs(f - r) > 3)
    comparable = (
        mismatches == 0
        and per_arm["fusion"]["errors"] == 0
        and (per_arm["reranked"]["errors"] == 0)
    )

    pool = {
        "fusion_mean_candidates": (
            round(statistics.mean(f_counts), 1) if f_counts else 0
        ),
        "reranked_mean_candidates": (
            round(statistics.mean(r_counts), 1) if r_counts else 0
        ),
        "questions_with_pool_mismatch": mismatches,
        "fusion_errors": per_arm["fusion"]["errors"],
        "reranked_errors": per_arm["reranked"]["errors"],
        "arms_comparable": comparable,
    }

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "golden_set_size": len(items),
        "golden_set_metadata": golden.get("metadata", {}),
        "config": {
            "retriever_k": app_config.RETRIEVER_K,
            "rerank_top_n": app_config.RERANK_TOP_N,
            "chunk_size": app_config.CHUNK_SIZE,
            "chunk_overlap": app_config.CHUNK_OVERLAP,
            "embedding_model": app_config.EMBEDDING_MODEL,
            "utility_model": app_config.UTILITY_MODEL,
            "interleaved": True,
            "delay_seconds": args.delay,
        },
        "comparability": pool,
        "arms": {
            arm: {"metrics": _summarise(d["ranks"], d["latencies"])}
            for arm, d in per_arm.items()
        },
        "per_question": per_question,
    }

    before = report["arms"]["fusion"]["metrics"]
    after = report["arms"]["reranked"]["metrics"]
    delta = {}
    for k in ev.K_VALUES:
        key = f"hit_rate@{k}"
        b, a = before[key], after[key]
        delta[key] = {
            "before": b,
            "after": a,
            "absolute": round(a - b, 4),
            "relative_pct": round(((a - b) / b) * 100, 1) if b else None,
        }
    delta["mrr"] = {
        "before": before["mrr"],
        "after": after["mrr"],
        "absolute": round(after["mrr"] - before["mrr"], 4),
    }
    report["rerank_ablation"] = delta

    print("\n" + "=" * 60)
    print("CANDIDATE POOL COMPARABILITY")
    print("=" * 60)
    print(f"  fusion   mean candidates: {pool['fusion_mean_candidates']}")
    print(f"  reranked mean candidates: {pool['reranked_mean_candidates']}")
    print(f"  questions with pool mismatch (>3): {mismatches}")
    print(
        f"  retrieval errors: fusion={pool['fusion_errors']} "
        f"reranked={pool['reranked_errors']}"
    )

    if not comparable:
        print("\n  *** ARMS NOT COMPARABLE ***")
        print("  Multi-Query/HyDE failed on some questions (rate limit or")
        print("  connection error), so the arms saw different candidate pools.")
        print("  The ablation below is CONFOUNDED. Re-run with a larger --delay")
        print("  before recording any of these numbers.")
    else:
        print("\n  Arms comparable. Ablation below is valid.")

    print("\n" + "=" * 60)
    print("RE-RANKING ABLATION" + ("" if comparable else "  [CONFOUNDED]"))
    print("=" * 60)
    for k in ev.K_VALUES:
        d = delta[f"hit_rate@{k}"]
        rel = f"{d['relative_pct']:+.1f}%" if d["relative_pct"] is not None else "n/a"
        print(f"  hit_rate@{k}: {d['before']:.3f} -> {d['after']:.3f}  ({rel})")
    print(f"  MRR:         {delta['mrr']['before']:.3f} -> {delta['mrr']['after']:.3f}")
    print("=" * 60)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ev.RESULTS_DIR / (args.out or f"retrieval_{stamp}.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if comparable:
        (ev.RESULTS_DIR / "retrieval_latest.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
    else:
        print("\nNOT written to retrieval_latest.json - confounded runs must not")
        print("become the baseline that thresholds are derived from.")
    print(f"\nWritten to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
