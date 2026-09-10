"""Stage 2: human review. This is the step that makes the set a GOLDEN set.

Every candidate is shown with its anchor passage. You accept, reject, or edit.
Nothing reaches golden_set.json without a decision from you.

What to reject:
  - Question is a thin paraphrase of a sentence in the passage
  - Question is unanswerable without seeing the passage ("what does Table 3 show")
  - Question is trivially generic ("what is physics")
  - Question is answerable from half the corpus (label would be meaningless)
  - Question is factually confused by the generator

Aim to accept ~65-75%. If you're accepting everything, you aren't reviewing.

Resumable: re-run and it picks up where you stopped.

Usage:
    python -m eval.review_golden_set
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

from eval import config as ev


def load_jsonl(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_golden():
    if not ev.GOLDEN_SET_PATH.exists():
        return {"items": [], "metadata": {}}
    return json.loads(ev.GOLDEN_SET_PATH.read_text(encoding="utf-8"))


def save_golden(golden, reviewed_count, rejected_count):
    golden["metadata"] = {
        "session_id": ev.EVAL_SESSION_ID,
        "size": len(golden["items"]),
        "construction": "chunk_anchored_generated_human_verified",
        "reviewed": reviewed_count,
        "rejected": rejected_count,
        "acceptance_rate": (
            round(len(golden["items"]) / reviewed_count, 3) if reviewed_count else None
        ),
        "label_type": "single anchor chunk per question (lower bound on recall)",
        "reference_answers": False,
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }
    ev.GOLDEN_SET_PATH.write_text(
        json.dumps(golden, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> int:
    candidates = load_jsonl(ev.CANDIDATES_PATH)
    if not candidates:
        print("No candidates. Run: python -m eval.build_golden_set")
        return 1

    golden = load_golden()
    done_ids = {item["id"] for item in golden["items"]}
    seen_path = ev.EVAL_DIR / ".reviewed_ids"
    seen = set(seen_path.read_text().split()) if seen_path.exists() else set()

    pending = [c for c in candidates if c["id"] not in seen]
    print(f"{len(pending)} to review. {len(golden['items'])} already accepted.\n")
    print("[a]ccept  [r]eject  [e]dit question  [q]uit and save\n")

    rejected = len(seen) - len(done_ids)

    for cand in pending:
        print("=" * 78)
        print(f"{cand['id']}  ·  {cand['source_doc']}  ·  page {cand['page']}")
        print("-" * 78)
        print(cand["anchor_text"][:900])
        print("-" * 78)
        print(f"Q: {cand['question']}")
        print()

        while True:
            choice = input("[a/r/e/q] > ").strip().lower()
            if choice == "a":
                cand["reviewed"] = True
                cand.pop("anchor_text", None)
                golden["items"].append(cand)
                seen.add(cand["id"])
                break
            if choice == "r":
                rejected += 1
                seen.add(cand["id"])
                break
            if choice == "e":
                edited = input("New question: ").strip()
                if edited:
                    cand["question"] = edited
                    cand["provenance"] = "chunk_anchored_generated_human_edited"
                    print(f"Q: {cand['question']}")
                continue
            if choice == "q":
                seen_path.write_text(" ".join(sorted(seen)))
                save_golden(golden, len(seen), rejected)
                print(f"\nSaved {len(golden['items'])} items to {ev.GOLDEN_SET_PATH}")
                return 0
            print("a / r / e / q")

        if len(golden["items"]) >= ev.TARGET_GOLDEN_SIZE:
            print(f"\nReached target of {ev.TARGET_GOLDEN_SIZE}.")
            break

    seen_path.write_text(" ".join(sorted(seen)))
    save_golden(golden, len(seen), rejected)
    print(f"\nGolden set: {len(golden['items'])} accepted, {rejected} rejected")
    print(f"Written to {ev.GOLDEN_SET_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
