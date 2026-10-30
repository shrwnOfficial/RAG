"""Ingest orchestration: hash, route, chunk, persist."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import config
from rag.ingest.chunker import chunk_document
from rag.ingest.excel import load_csv, load_excel
from rag.ingest.ocr import ocr_bytes
from rag.ingest.pdf import load_pdf
from rag.store import DocumentStore


ProgressCallback = Callable[[str, float], None]
PageProgressCallback = Callable[[int, int], None]


@dataclass
class IngestResult:
    filename: str
    doc_id: str
    skipped: bool
    chunk_count: int
    page_count: int


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _load_registry() -> dict:
    if not config.REGISTRY_PATH.exists():
        return {"files": {}}
    return json.loads(config.REGISTRY_PATH.read_text(encoding="utf-8"))


def _save_registry(registry: dict) -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.REGISTRY_PATH.write_text(
        json.dumps(registry, indent=2),
        encoding="utf-8",
    )


def list_indexed_documents() -> list[dict]:
    registry = _load_registry()
    return list(registry.get("files", {}).values())


def delete_document(doc_id: str, store: DocumentStore) -> None:
    registry = _load_registry()
    to_remove = None
    for sha, meta in registry.get("files", {}).items():
        if meta.get("doc_id") == doc_id:
            to_remove = sha
            upload_path = config.UPLOADS_DIR / meta.get("stored_name", "")
            if upload_path.exists():
                upload_path.unlink()
            break
    if to_remove:
        del registry["files"][to_remove]
        _save_registry(registry)
    store.delete_by_doc_id(doc_id)


def clear_all(store: DocumentStore) -> None:
    if config.UPLOADS_DIR.exists():
        shutil.rmtree(config.UPLOADS_DIR)
    config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    _save_registry({"files": {}})
    store.clear()


def ingest_file(
    path: Path,
    store: DocumentStore,
    on_progress: ProgressCallback | None = None,
    on_page: PageProgressCallback | None = None,
) -> IngestResult:
    path = Path(path)
    filename = path.name
    file_hash = _file_sha256(path)
    registry = _load_registry()

    if file_hash in registry.get("files", {}):
        meta = registry["files"][file_hash]
        return IngestResult(
            filename=meta["filename"],
            doc_id=meta["doc_id"],
            skipped=True,
            chunk_count=meta.get("chunk_count", 0),
            page_count=meta.get("page_count", 0),
        )

    doc_id = file_hash[:16]
    config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    stored_name = f"{doc_id}_{filename}"
    dest = config.UPLOADS_DIR / stored_name
    shutil.copy2(path, dest)

    suffix = path.suffix.lower()
    pages: list[tuple[int, str, str]] = []
    doc_level_parts: list[str] = []

    if on_progress:
        on_progress(f"Reading {filename}…", 0.1)

    if suffix == ".pdf":

        def page_cb(cur: int, total: int) -> None:
            if on_page:
                on_page(cur, total)
            if on_progress:
                on_progress(f"Processing page {cur}/{total}…", 0.1 + 0.5 * (cur / max(total, 1)))

        pdf_pages = load_pdf(dest, on_page=page_cb)
        for p in pdf_pages:
            pages.append((p.page_num, p.text, p.source_type))
            doc_level_parts.append(p.text)
    elif suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        text = ocr_bytes(dest.read_bytes())
        pages.append((1, text, "ocr"))
        doc_level_parts.append(text)
    elif suffix == ".csv":
        for sheet in load_csv(dest):
            pages.append((sheet.page_num, sheet.text, sheet.source_type))
            doc_level_parts.append(sheet.text)
    elif suffix in (".xlsx", ".xls"):
        for sheet in load_excel(dest):
            pages.append((sheet.page_num, sheet.text, sheet.source_type))
            doc_level_parts.append(sheet.text)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    if on_progress:
        on_progress("Chunking…", 0.65)

    chunks = chunk_document(
        doc_id=doc_id,
        filename=filename,
        pages=pages,
        doc_level_text="\n".join(doc_level_parts),
    )

    if not chunks:
        raise ValueError(f"No text extracted from {filename}")

    if on_progress:
        on_progress("Embedding and indexing…", 0.8)

    store.add_chunks(chunks)

    meta = {
        "filename": filename,
        "doc_id": doc_id,
        "file_hash": file_hash,
        "stored_name": stored_name,
        "chunk_count": len(chunks),
        "page_count": len(pages),
    }
    registry.setdefault("files", {})[file_hash] = meta
    _save_registry(registry)

    if on_progress:
        on_progress("Done", 1.0)

    return IngestResult(
        filename=filename,
        doc_id=doc_id,
        skipped=False,
        chunk_count=len(chunks),
        page_count=len(pages),
    )
