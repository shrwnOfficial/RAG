"""Application configuration (override via environment variables)."""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("RAG_DATA_DIR", str(BASE_DIR / "data")))
CHROMA_DIR = DATA_DIR / "chroma"
REGISTRY_PATH = DATA_DIR / "registry.json"
UPLOADS_DIR = DATA_DIR / "uploads"

EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
RERANKER_MODEL = os.getenv("RAG_RERANKER_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
GEMINI_LITE_MODEL = os.getenv("GEMINI_LITE_MODEL", "gemini-3.5-flash-lite")

# Approximate token limits (~4 chars per token for English).
CHUNK_TARGET_CHARS = int(os.getenv("RAG_CHUNK_TARGET_CHARS", "1400"))
CHUNK_OVERLAP_CHARS = int(os.getenv("RAG_CHUNK_OVERLAP_CHARS", "200"))
CHUNK_MAX_CHARS = int(os.getenv("RAG_CHUNK_MAX_CHARS", "1800"))

TOP_K_VECTOR = int(os.getenv("RAG_TOP_K_VECTOR", "20"))
TOP_K_BM25 = int(os.getenv("RAG_TOP_K_BM25", "20"))
TOP_K_FINAL = int(os.getenv("RAG_TOP_K_FINAL", "6"))
RRF_K = int(os.getenv("RAG_RRF_K", "60"))

USE_RERANKER = os.getenv("RAG_USE_RERANKER", "true").lower() in ("1", "true", "yes")
OCR_DPI = int(os.getenv("RAG_OCR_DPI", "300"))
OCR_MIN_CHARS = int(os.getenv("RAG_OCR_MIN_CHARS", "50"))
OCR_MIN_ALNUM_RATIO = float(os.getenv("RAG_OCR_MIN_ALNUM_RATIO", "0.6"))
TESSERACT_LANG = os.getenv("TESSERACT_LANG", "eng")

MAX_UPLOAD_MB = int(os.getenv("RAG_MAX_UPLOAD_MB", "50"))
REWRITE_HISTORY_TURNS = int(os.getenv("RAG_REWRITE_HISTORY_TURNS", "4"))

CHROMA_COLLECTION = "document_chunks"

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".xlsx",
    ".xls",
    ".csv",
}
