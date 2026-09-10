"""Stage 1 of golden-set construction: generate CANDIDATE questions.

Methodology — chunk-anchored generation with human verification.

  1. Sample N chunks at random (fixed seed) from the eval corpus.
  2. For each chunk, an LLM writes a question that the chunk answers.
  3. Because the question was written *from* that chunk, the chunk is by
     construction a relevant passage for it. That is the ground-truth label.
  4. A human then reviews every candidate and rejects the bad ones
     (see review_golden_set.py). Nothing enters the golden set unreviewed.

Why this is legitimate and what its limits are — state both in EVALUATION.md:

  LEGITIMATE: the label is not a model's *opinion* about relevance. It is a
  structural fact about how the question was produced. The model is used as a
  question writer, not as a judge.

  LIMIT: questions are phrased from the chunk's own vocabulary, which biases the
  set toward lexically-similar retrieval and will FLATTER dense retrieval
  relative to real student questions. The review step exists partly to fight
  this - reject anything that reads like a paraphrase of the source sentence.

  LIMIT: a chunk labelled relevant does not mean other chunks are irrelevant.
  Recall against a single known-relevant chunk is therefore a LOWER BOUND on
  true recall. Report it as such; do not call it precision.

Usage:
    python -m eval.build_golden_set
"""

from __future__ import annotations

import json
import random
import sys

from langchain_core.prompts import PromptTemplate

from eval import config as ev
from app.core import config as app_config
from app.services import ingestion
from app.services.retrieval import get_utility_llm

QUESTION_PROMPT = PromptTemplate(
    input_variables=["passage"],
    template=(
        "You are helping build an evaluation set for a study assistant.\n\n"
        "Below is a passage from a student's course material. Write ONE exam-style "
        "question that this passage answers.\n\n"
        "Rules:\n"
        "- Write it the way a student would ask it, in their own words.\n"
        "- Do NOT reuse distinctive phrases verbatim from the passage.\n"
        "- Do NOT reference 'the passage', 'the text', figures or page numbers.\n"
        "- It must be answerable from this passage alone.\n"
        "- One sentence. Output the question only, nothing else.\n\n"
        "PASSAGE:\n{passage}\n\nQUESTION:"
    ),
)


def chunk_key(metadata: dict) -> str:
    """Must stay identical to retrieval_eval.chunk_key — labels are matched by
    string equality, so any divergence silently produces 0.0 hit-rate."""
    return (
        f"{metadata.get('doc_id')}"
        f"::{metadata.get('page')}"
        f"::{metadata.get('chunk_index')}"
    )


def main() -> int:
    chunks = ingestion.get_all_chunks(ev.EVAL_SESSION_ID)
    if not chunks:
        print(f"No chunks for session {ev.EVAL_SESSION_ID}. Run eval.setup_corpus first.")
        return 1

    usable = [c for c in chunks if len(c.page_content.strip()) >= ev.MIN_CHUNK_CHARS]
    print(f"{len(chunks)} chunks total, {len(usable)} above {ev.MIN_CHUNK_CHARS} chars")

    if len(usable) < ev.CANDIDATE_COUNT:
        print(f"Only {len(usable)} usable chunks; sampling all of them.")

    rng = random.Random(ev.RANDOM_SEED)
    sample = rng.sample(usable, min(ev.CANDIDATE_COUNT, len(usable)))

    llm = get_utility_llm()
    chain = QUESTION_PROMPT | llm

    written = 0
    with ev.CANDIDATES_PATH.open("w", encoding="utf-8") as fh:
        for i, chunk in enumerate(sample, start=1):
            try:
                question = chain.invoke({"passage": chunk.page_content}).content.strip()
            except Exception as exc:  # noqa: BLE001
                print(f"  [{i}] generation failed: {exc}")
                continue

            question = question.strip().strip('"')
            if not question or len(question) < 15:
                print(f"  [{i}] rejected: degenerate output")
                continue

            record = {
                "id": f"q{i:03d}",
                "question": question,
                "relevant_chunk_ids": [chunk_key(chunk.metadata)],
                "anchor_text": chunk.page_content,
                "source_doc": chunk.metadata.get("source"),
                "page": chunk.metadata.get("page"),
                "provenance": "chunk_anchored_generated",
                "reviewed": False,
                "reference_answer": None,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            print(f"  [{i}] {question[:80]}")

    print(f"\n{written} candidates written to {ev.CANDIDATES_PATH}")
    print("NEXT: python -m eval.review_golden_set")
    print(f"Generator model: {app_config.UTILITY_MODEL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
