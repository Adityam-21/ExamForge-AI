"""Evaluation configuration.

Single source of truth for paths, the eval corpus session and the judge model.
Every number this harness produces is recorded alongside these values, because
a metric without its configuration is not reproducible.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Paths -------------------------------------------------------------------

EVAL_DIR = Path(__file__).parent
RESULTS_DIR = EVAL_DIR / "results"
CANDIDATES_PATH = EVAL_DIR / "candidates.jsonl"
GOLDEN_SET_PATH = EVAL_DIR / "golden_set.json"
CORPUS_DIR = EVAL_DIR / "corpus"          # put the eval PDFs here

RESULTS_DIR.mkdir(exist_ok=True)
CORPUS_DIR.mkdir(exist_ok=True)

# --- Eval corpus -------------------------------------------------------------

# A fixed, deterministic session id. The eval corpus is ingested once into this
# session and never mixed with real user sessions, so retrieval results are
# reproducible across runs and machines.
EVAL_SESSION_ID = os.getenv("EVAL_SESSION_ID", "eval-corpus-v1")

# --- Golden set construction -------------------------------------------------

# Chunks sampled to anchor questions. Over-sample: human review rejects some.
CANDIDATE_COUNT = int(os.getenv("EVAL_CANDIDATE_COUNT", "60"))
TARGET_GOLDEN_SIZE = int(os.getenv("EVAL_TARGET_SIZE", "40"))
MIN_CHUNK_CHARS = 300      # skip near-empty chunks (headers, page numbers)
RANDOM_SEED = 20260910     # fixed so the sample is reproducible

# --- Metrics -----------------------------------------------------------------

K_VALUES = (1, 3, 5)

# --- RAGAS -------------------------------------------------------------------

# RAGAS uses an LLM as judge. The judge is pinned and recorded with every result;
# scores from different judges are not comparable.
JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "openai/gpt-oss-120b")
RAGAS_RUNS = int(os.getenv("EVAL_RAGAS_RUNS", "3"))
RAGAS_SAMPLE_SIZE = int(os.getenv("EVAL_RAGAS_SAMPLE", "20"))

# --- Fast tier ---------------------------------------------------------------

FAST_TIER_SIZE = int(os.getenv("EVAL_FAST_TIER_SIZE", "12"))
