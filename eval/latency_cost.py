"""D4 - re-verify end-to-end latency and per-query cost.

Measures the real pipeline (condense -> retrieve -> generate) over the golden
set and reports the full distribution, not just a headline.

Method is recorded in the output so the number can be defended:
  - sample size
  - cold vs warm (first run is discarded as cold; stated explicitly)
  - which models
  - what is and is not included

TOKEN COSTS: this script records what the Groq client reports in
`usage_metadata` where available. If your generation path streams without
surfacing usage, the token counts come from the LangSmith traces instead - that
is one of the reasons D1 comes first. Do not fill in an estimate.

Usage:
    python -m eval.latency_cost --n 30
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
from app.services.agent import run_pipeline


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    # Nearest-rank. Stated in the output so the method is unambiguous.
    index = max(0, min(len(ordered) - 1, int(round(p / 100 * len(ordered) + 0.5)) - 1))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30)
    args = parser.parse_args()

    golden = json.loads(ev.GOLDEN_SET_PATH.read_text(encoding="utf-8"))
    items = golden["items"][: args.n]

    print(f"Timing {len(items)} questions (first discarded as cold)\n")

    samples = []
    cold_ms = None

    for i, item in enumerate(items, start=1):
        started = time.perf_counter()
        try:
            result = run_pipeline(ev.EVAL_SESSION_ID, item["question"], [])
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i}] FAILED: {exc}")
            continue
        elapsed = time.perf_counter() - started

        if i == 1:
            cold_ms = round(elapsed * 1000, 1)
            print(f"  [  1] COLD  {elapsed:.2f}s  (discarded)")
            continue

        samples.append(elapsed)
        answer_chars = len(result.get("answer") or "")
        print(f"  [{i:>3}] {elapsed:.2f}s  {answer_chars} chars  "
              f"{len(result.get('citations') or [])} sources")

    if not samples:
        print("No warm samples collected.")
        return 1

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "method": {
            "sample_size_warm": len(samples),
            "cold_start_ms": cold_ms,
            "cold_excluded": True,
            "percentile_method": "nearest-rank",
            "measured": "end-to-end run_pipeline: condense + retrieve "
                        "(4 strategies) + rerank + generate, streaming to completion",
            "not_measured": "HTTP overhead, SSE transport, frontend render, "
                            "network latency from the user",
            "environment": "local / CI - record which when you paste this",
            "generation_model": app_config.GENERATION_MODEL,
            "utility_model": app_config.UTILITY_MODEL,
            "retriever_k": app_config.RETRIEVER_K,
            "rerank_top_n": app_config.RERANK_TOP_N,
        },
        "latency_seconds": {
            "min": round(min(samples), 3),
            "p50": round(percentile(samples, 50), 3),
            "p90": round(percentile(samples, 90), 3),
            "p95": round(percentile(samples, 95), 3),
            "p99": round(percentile(samples, 99), 3),
            "max": round(max(samples), 3),
            "mean": round(statistics.mean(samples), 3),
            "stdev": round(statistics.stdev(samples), 3) if len(samples) > 1 else None,
        },
        "cost": {
            "note": "Token counts not captured here. Read them from the "
                    "LangSmith project (D1) and record them, or leave null. "
                    "Do not estimate.",
            "input_tokens_per_query": None,
            "output_tokens_per_query": None,
            "inr_per_query": None,
        },
    }

    print("\n" + "=" * 60)
    print("LATENCY (warm, end-to-end pipeline)")
    print("=" * 60)
    for key, value in report["latency_seconds"].items():
        print(f"  {key:>6}: {value}s" if value is not None else f"  {key:>6}: n/a")
    print(f"\n  cold start: {cold_ms}ms")
    print("=" * 60)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ev.RESULTS_DIR / f"latency_{stamp}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (ev.RESULTS_DIR / "latency_latest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"\nWritten to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
