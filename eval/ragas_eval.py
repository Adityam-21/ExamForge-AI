"""D2 - RAGAS generation-side metrics. Pinned to the installed RAGAS 0.1.19 API.

WHICH METRICS THIS GOLDEN SET SUPPORTS
--------------------------------------
Anchor-chunk labels, no reference answers. That means:

  RUNNABLE (no ground truth needed)
    faithfulness      - is every claim in the answer supported by the retrieved
                        context? The hallucination metric, and the one that
                        matters most for a study assistant.
    answer_relevancy  - does the answer address the question asked?

  NOT RUNNABLE (needs reference answers)
    context_recall, answer_correctness, answer_similarity

Retrieval precision is already measured deterministically in retrieval_eval.py,
which is cheaper and has no judge variance. Don't pay an LLM to redo it.

VARIANCE
--------
The judge is an LLM, so one run is not a measurement. This runs the suite
EVAL_RAGAS_RUNS times and reports mean, range and spread. Quote the range.

Usage:
    python -m eval.ragas_eval
"""

from __future__ import annotations

import json
import statistics
import sys
from datetime import datetime, timezone

from eval import config as ev
from app.core import config as app_config
from app.services.agent import run_pipeline


def build_rows(items):
    """Run the real pipeline and collect (question, answer, contexts) triples."""
    rows, abstained = [], 0
    for i, item in enumerate(items, start=1):
        try:
            result = run_pipeline(ev.EVAL_SESSION_ID, item["question"], [])
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i}] pipeline failed: {exc}")
            continue

        contexts = [c["text"] for c in (result.get("citations") or []) if c.get("text")]
        if not contexts:
            # Abstention: no context to be faithful to. Excluded from scoring,
            # counted separately - abstention rate is its own metric.
            print(f"  [{i}] abstained (excluded)")
            abstained += 1
            continue

        rows.append(
            {
                "question": item["question"],
                "answer": result["answer"],
                "contexts": contexts,
            }
        )
        print(f"  [{i}] collected ({len(contexts)} contexts)")
    return rows, abstained


def main() -> int:
    from datasets import Dataset
    from langchain_groq import ChatGroq
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import answer_relevancy, faithfulness

    from app.services.ingestion import embedding_model

    golden = json.loads(ev.GOLDEN_SET_PATH.read_text(encoding="utf-8"))
    items = golden["items"][: ev.RAGAS_SAMPLE_SIZE]

    print(f"Collecting pipeline outputs for {len(items)} questions...")
    rows, abstained = build_rows(items)
    total = len(rows) + abstained
    print(
        f"\n{len(rows)} scorable, {abstained} abstentions "
        f"(rate {abstained / max(total, 1):.1%})\n"
    )

    if not rows:
        print("Nothing to score.")
        return 1

    judge = LangchainLLMWrapper(
        ChatGroq(model=ev.JUDGE_MODEL, temperature=0, api_key=app_config.GROQ_API_KEY)
    )
    # answer_relevancy embeds generated questions in RAGAS 0.1.x. Reusing the
    # app's local BGE model keeps this off the API and off the token budget.
    embeddings = LangchainEmbeddingsWrapper(embedding_model)

    dataset = Dataset.from_dict(
        {
            "question": [r["question"] for r in rows],
            "answer": [r["answer"] for r in rows],
            "contexts": [r["contexts"] for r in rows],
        }
    )

    runs = []
    for run_index in range(1, ev.RAGAS_RUNS + 1):
        print(f"--- RAGAS run {run_index}/{ev.RAGAS_RUNS} ---")
        try:
            result = evaluate(
                dataset=dataset,
                metrics=[faithfulness, answer_relevancy],
                llm=judge,
                embeddings=embeddings,
                raise_exceptions=False,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  run failed: {exc}")
            continue
        scores = {k: round(float(v), 4) for k, v in dict(result).items()}
        print(f"  {scores}")
        runs.append(scores)

    if not runs:
        print("All runs failed.")
        return 1

    summary = {}
    for key in sorted({k for run in runs for k in run}):
        values = [run[key] for run in runs if key in run]
        summary[key] = {
            "runs": values,
            "mean": round(statistics.mean(values), 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
            "spread": round(max(values) - min(values), 4),
            "stdev": round(statistics.stdev(values), 4) if len(values) > 1 else None,
        }

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ragas_version": "0.1.19",
        "judge_model": ev.JUDGE_MODEL,
        "judge_temperature": 0,
        "embedding_model": app_config.EMBEDDING_MODEL,
        "generation_model": app_config.GENERATION_MODEL,
        "rerank_enabled": getattr(app_config, "RERANK_ENABLED", None),
        "n_questions_scored": len(rows),
        "n_abstained": abstained,
        "abstention_rate": round(abstained / max(total, 1), 4),
        "run_count": len(runs),
        "metrics": summary,
        "metrics_not_run": {
            "context_recall": "requires reference answers - not available",
            "answer_correctness": "requires reference answers - not available",
        },
    }

    print("\n" + "=" * 62)
    print(f"RAGAS  judge={ev.JUDGE_MODEL}  runs={len(runs)}  n={len(rows)}")
    print("=" * 62)
    for key, s in summary.items():
        print(
            f"  {key:<20} {s['mean']:.3f}  "
            f"(range {s['min']:.3f}-{s['max']:.3f}, spread {s['spread']:.3f})"
        )
    print("=" * 62)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (ev.RESULTS_DIR / f"ragas_{stamp}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (ev.RESULTS_DIR / "ragas_latest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"\nWritten to eval/results/ragas_{stamp}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
