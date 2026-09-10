"""Deterministic retrieval evaluation with a re-ranking ablation.

This is the script that produces the honest replacement for the hit-rate claim.

No LLM judge is involved: a chunk either appears in the top-k or it does not.
The numbers are reproducible, cheap, and defensible under questioning.

THE ABLATION
------------
Two arms, identical in every respect except one:

  arm "fusion"   - retrieval runs, rerank() is replaced by an identity function
                   that truncates to the same depth. Ranking is first-stage
                   fusion order (RRF over dense + BM25, plus Multi-Query/HyDE).
  arm "reranked" - production path, FlashRank cross-encoder reordering.

The ablation is done by monkeypatching `retrieval.rerank` inside this script.
NO SOURCE FILE IS MODIFIED. That matters: the production code path measured in
the "reranked" arm is byte-identical to what runs in production.

METRICS
-------
  hit_rate@k  - fraction of questions where the anchor chunk is in the top k
  mrr         - mean reciprocal rank of the anchor chunk (0 if absent)
  recall@k    - identical to hit_rate@k here, since each question has exactly
                one labelled chunk. Reported once, named honestly.

Usage:
    python -m eval.retrieval_eval
    python -m eval.retrieval_eval --arms reranked        # single arm
    python -m eval.retrieval_eval --limit 12             # fast tier
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

# Keep a handle on the real reranker before anything patches it.
_REAL_RERANK = retrieval.rerank


def _identity_rerank(question, chunks):
    """First-stage fusion order, truncated to the same depth as the reranker.

    Depth is matched deliberately. Comparing top-5 reranked against top-20 fused
    would measure truncation, not re-ranking.
    """
    return chunks[: app_config.RERANK_TOP_N]


def chunk_key(metadata: dict) -> str:
    return f"{metadata.get('doc_id')}::{metadata.get('chunk_index')}"


def rank_of_anchor(results, relevant_ids: set[str]) -> int | None:
    for position, chunk in enumerate(results, start=1):
        if chunk_key(chunk.metadata or {}) in relevant_ids:
            return position
    return None


def run_arm(name: str, items: list[dict]) -> dict:
    """Evaluate one arm over the whole golden set."""
    if name == "fusion":
        retrieval.rerank = _identity_rerank
    else:
        retrieval.rerank = _REAL_RERANK

    ranks: list[int | None] = []
    latencies: list[float] = []
    per_question = []

    for i, item in enumerate(items, start=1):
        relevant = set(item["relevant_chunk_ids"])
        started = time.perf_counter()
        try:
            results = retrieval.retrieve(ev.EVAL_SESSION_ID, item["question"])
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i}] {item['id']} FAILED: {exc}")
            ranks.append(None)
            per_question.append({"id": item["id"], "rank": None, "error": str(exc)})
            continue
        elapsed = time.perf_counter() - started

        rank = rank_of_anchor(results, relevant)
        ranks.append(rank)
        latencies.append(elapsed)
        per_question.append(
            {
                "id": item["id"],
                "rank": rank,
                "retrieved": len(results),
                "retrieval_seconds": round(elapsed, 3),
            }
        )
        marker = f"@{rank}" if rank else "MISS"
        print(f"  [{i:>3}] {item['id']}  {marker:>5}  {elapsed:.2f}s")

    n = len(ranks)
    metrics = {
        f"hit_rate@{k}": round(
            sum(1 for r in ranks if r is not None and r <= k) / n, 4
        )
        for k in ev.K_VALUES
    }
    metrics["mrr"] = round(
        sum((1 / r) if r else 0.0 for r in ranks) / n, 4
    )
    metrics["n_questions"] = n
    metrics["retrieval_p50_seconds"] = (
        round(statistics.median(latencies), 3) if latencies else None
    )

    retrieval.rerank = _REAL_RERANK  # always restore
    return {"metrics": metrics, "per_question": per_question}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", default=["fusion", "reranked"])
    parser.add_argument("--limit", type=int, default=None)
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
    print(f"Config: k={app_config.RETRIEVER_K} rerank_top_n={app_config.RERANK_TOP_N} "
          f"chunk={app_config.CHUNK_SIZE}/{app_config.CHUNK_OVERLAP}\n")

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
        },
        "arms": {},
    }

    for arm in args.arms:
        print(f"--- arm: {arm} ---")
        report["arms"][arm] = run_arm(arm, items)
        print()

    # Ablation delta, only when both arms ran.
    if "fusion" in report["arms"] and "reranked" in report["arms"]:
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

        print("=" * 60)
        print("RE-RANKING ABLATION")
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
    (ev.RESULTS_DIR / "retrieval_latest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"\nWritten to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
