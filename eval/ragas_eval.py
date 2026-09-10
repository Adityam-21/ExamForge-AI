"""D2 - RAGAS generation-side metrics.

WHICH METRICS YOUR GOLDEN SET SUPPORTS
--------------------------------------
Your set has anchor-chunk labels and NO reference answers. That puts you here:

  RUNNABLE NOW (no ground truth needed)
    faithfulness      - is every claim in the answer supported by the retrieved
                        context? This is the hallucination metric and it is the
                        one that matters most for a study assistant.
    answer_relevancy  - does the answer actually address the question asked?

  NEEDS REFERENCE ANSWERS (not runnable yet)
    context_recall    - did retrieval find everything the reference answer needs?
    answer_correctness- is the answer factually right?

  Retrieval precision is already covered deterministically by retrieval_eval.py,
  which is cheaper and has no judge variance. Don't pay an LLM to re-measure it.

If you later want the ground-truth metrics: write reference answers by hand for
20-30 questions, from the anchor chunk, in your own words. Do NOT generate them
with a model and call them ground truth - a model-written reference judged by a
model is a closed loop that measures nothing.

VARIANCE
--------
RAGAS uses an LLM as judge, so a single run is not a measurement. This script
runs the suite RAGAS_RUNS times and reports mean and spread. Quote the range.

API NOTE: RAGAS changes its public API between minor versions. Pin it
(`ragas==0.2.x`) and if the imports below fail, check the installed version's
docs rather than guessing - the concepts are stable even when names move.

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


def build_dataset(items):
    """Run the real pipeline and collect (question, answer, contexts) triples."""
    rows = []
    for i, item in enumerate(items, start=1):
        try:
            result = run_pipeline(ev.EVAL_SESSION_ID, item["question"], [])
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i}] pipeline failed: {exc}")
            continue

        contexts = [c["text"] for c in (result.get("citations") or []) if c.get("text")]
        if not contexts:
            # Abstention. Excluded from faithfulness - there is no context to be
            # faithful to. Counted separately; abstention rate is its own metric.
            print(f"  [{i}] abstained (excluded from RAGAS)")
            rows.append({"_abstained": True})
            continue

        rows.append(
            {
                "user_input": item["question"],
                "response": result["answer"],
                "retrieved_contexts": contexts,
            }
        )
        print(f"  [{i}] collected ({len(contexts)} contexts)")
    return rows


def main() -> int:
    from langchain_groq import ChatGroq
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, ResponseRelevancy

    golden = json.loads(ev.GOLDEN_SET_PATH.read_text(encoding="utf-8"))
    items = golden["items"][: ev.RAGAS_SAMPLE_SIZE]

    print(f"Collecting pipeline outputs for {len(items)} questions...")
    rows = build_dataset(items)

    abstained = sum(1 for r in rows if r.get("_abstained"))
    scored = [r for r in rows if not r.get("_abstained")]
    print(f"\n{len(scored)} scorable, {abstained} abstentions "
          f"(abstention rate {abstained / max(len(rows), 1):.1%})\n")

    if not scored:
        print("Nothing to score.")
        return 1

    judge = LangchainLLMWrapper(
        ChatGroq(
            model=ev.JUDGE_MODEL,
            temperature=0,
            api_key=app_config.GROQ_API_KEY,
        )
    )
    metrics = [Faithfulness(llm=judge), ResponseRelevancy(llm=judge)]
    dataset = EvaluationDataset.from_list(scored)

    runs = []
    for run_index in range(1, ev.RAGAS_RUNS + 1):
        print(f"--- RAGAS run {run_index}/{ev.RAGAS_RUNS} ---")
        result = evaluate(dataset=dataset, metrics=metrics, llm=judge)
        scores = {k: round(float(v), 4) for k, v in result._repr_dict.items()} \
            if hasattr(result, "_repr_dict") else \
            {k: round(float(v), 4) for k, v in dict(result).items()}
        print(f"  {scores}")
        runs.append(scores)

    keys = sorted({k for run in runs for k in run})
    summary = {}
    for key in keys:
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
        "judge_model": ev.JUDGE_MODEL,
        "judge_temperature": 0,
        "generation_model": app_config.GENERATION_MODEL,
        "n_questions_scored": len(scored),
        "n_abstained": abstained,
        "abstention_rate": round(abstained / max(len(rows), 1), 4),
        "run_count": ev.RAGAS_RUNS,
        "metrics": summary,
        "metrics_not_run": {
            "context_recall": "requires reference answers - not available",
            "answer_correctness": "requires reference answers - not available",
        },
    }

    print("\n" + "=" * 60)
    print(f"RAGAS  ·  judge={ev.JUDGE_MODEL}  ·  {ev.RAGAS_RUNS} runs")
    print("=" * 60)
    for key, stats in summary.items():
        print(f"  {key:<22} {stats['mean']:.3f}  "
              f"(range {stats['min']:.3f}-{stats['max']:.3f}, "
              f"spread {stats['spread']:.3f})")
    print("=" * 60)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ev.RESULTS_DIR / f"ragas_{stamp}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (ev.RESULTS_DIR / "ragas_latest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"\nWritten to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
