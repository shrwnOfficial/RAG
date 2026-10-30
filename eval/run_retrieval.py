"""Run retrieval eval against questions.jsonl (no LLM calls)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.ingest.loader import clear_all, ingest_file
from rag.retrieve import retrieve
from rag.store import DocumentStore

QUESTIONS = ROOT / "eval" / "questions.jsonl"
DOCS = ROOT / "docs"


EVAL_DOCS = [
    "06_MPCB_Consent_to_Operate_01.pdf",
    "09_CGWA_Groundwater_NOC_Beverage_Plant.pdf",
    "08_TNPCB_Proceedings_Consent_Sugar_Mill_Cogeneration.pdf",
]


def ensure_eval_index(store: DocumentStore) -> None:
    for name in EVAL_DOCS:
        path = DOCS / name
        if path.exists():
            ingest_file(path, store)


def main() -> int:
    store = DocumentStore()
    if store.chunk_count() == 0:
        print("Building eval index (3 sample PDFs)…")
        clear_all(store)
        ensure_eval_index(store)

    passed = 0
    total = 0
    for line in QUESTIONS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        question = row["question"]
        expected = row.get("expected_contains", [])
        total += 1
        result = retrieve(store, question, [])
        blob = "\n".join(c.text for c in result.chunks)
        ok = all(exp.lower() in blob.lower() for exp in expected)
        passed += int(ok)
        status = "PASS" if ok else "FAIL"
        print(f"{status}: {question}")
    print(f"\n{passed}/{total} retrieval checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
