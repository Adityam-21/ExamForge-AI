"""Ingest the evaluation corpus into a fixed session.

Run once. Every downstream script queries EVAL_SESSION_ID, so retrieval results
are reproducible: same documents, same chunking, same collection.

Usage:
    # drop 3-5 PDFs into eval/corpus/ first
    python -m eval.setup_corpus
"""

from __future__ import annotations

import hashlib
import json
import sys

from eval import config
from app.services import ingestion


def main() -> int:
    pdfs = sorted(config.CORPUS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs in {config.CORPUS_DIR}. Add 3-5 and re-run.")
        return 1

    manifest: list[dict] = []
    skipped: list[dict] = []

    for path in pdfs:
        # Deterministic doc_id from file content, so re-ingesting the same
        # corpus produces the same ids and the golden set stays valid.
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        print(f"Ingesting {path.name} (doc_id={digest}) ...")

        try:
            result = ingestion.ingest_pdf(
                str(path),
                config.EVAL_SESSION_ID,
                original_filename=path.name,
                doc_id=digest,
            )
        except Exception as exc:  # noqa: BLE001
            # One unparseable PDF must not abort the corpus build. Skipped files
            # are recorded so the corpus stays honestly described.
            print(f"  SKIPPED - {type(exc).__name__}: {str(exc)[:120]}")
            skipped.append({"file": path.name, "error": f"{type(exc).__name__}: {exc}"})
            continue

        print(
            f"  pages={result.get('page_count')} chunks={result.get('chunks_stored')}"
        )
        manifest.append(
            {
                "file": path.name,
                "doc_id": digest,
                "sha256_16": digest,
                "page_count": result.get("page_count"),
                "chunks_stored": result.get("chunks_stored"),
            }
        )

    chunks = ingestion.get_all_chunks(config.EVAL_SESSION_ID)
    print(f"\nTotal chunks in eval collection: {len(chunks)}")

    out = config.RESULTS_DIR / "corpus_manifest.json"
    out.write_text(
        json.dumps(
            {
                "session_id": config.EVAL_SESSION_ID,
                "documents": manifest,
                "total_chunks": len(chunks),
                "skipped": skipped,
            },
            indent=2,
        )
    )
    print(f"Manifest written to {out}")

    if skipped:
        print(f"\n{len(skipped)} file(s) skipped:")
        for entry in skipped:
            print(f"  - {entry['file']}")
        print("Recorded in the manifest so the corpus stays accurately described.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
