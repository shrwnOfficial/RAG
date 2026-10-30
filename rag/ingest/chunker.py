"""Structure-aware chunking with context headers."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

import config

SECTION_PATTERN = re.compile(
    r"^(?:"
    r"\d+\.\d*(?:\s|$)|"
    r"\d+\.\s|"
    r"\(\w+\)\s|"
    r"[A-Z]\.\s|"
    r"[IVX]+\.\s|"
    r"(?:SPECIAL|GENERAL)\s+CONDITIONS?"
    r")",
    re.IGNORECASE | re.MULTILINE,
)

COMPANY_PATTERNS = [
    re.compile(r"M/s\s+([^\n,]+)", re.IGNORECASE),
    re.compile(r"Name of the industry\s*:\s*([^\n]+)", re.IGNORECASE),
    re.compile(r"APPLICANT\s*\n\s*([^\n]+)", re.IGNORECASE),
]

REF_PATTERNS = [
    re.compile(r"Consent No\.?\s*([A-Za-z0-9\-/]+)", re.IGNORECASE),
    re.compile(r"Proceedings No\.?\s*([A-Za-z0-9\-/]+)", re.IGNORECASE),
    re.compile(r"NOC NO\.?\s*\n?\s*([A-Za-z0-9\-/]+)", re.IGNORECASE),
    re.compile(r"NOC No\.?\s*([A-Za-z0-9\-/]+)", re.IGNORECASE),
    re.compile(r"Licence No\.?\s*([A-Za-z0-9\-/]+)", re.IGNORECASE),
]


@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    filename: str
    page: int
    section: str
    text: str
    source_type: str
    company: str
    ref_no: str


def extract_company(full_text: str) -> str:
    for pat in COMPANY_PATTERNS:
        m = pat.search(full_text)
        if m:
            return m.group(1).strip()[:200]
    return ""


def extract_ref_no(full_text: str) -> str:
    for pat in REF_PATTERNS:
        m = pat.search(full_text)
        if m:
            return m.group(1).strip()[:120]
    return ""


def _split_sections(page_text: str) -> list[tuple[str, str]]:
    """Return list of (section_title, body)."""
    text = page_text.strip()
    if not text:
        return []

    matches = list(SECTION_PATTERN.finditer(text))
    if not matches:
        return [("", text)]

    sections: list[tuple[str, str]] = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end].strip()
        first_line = block.split("\n", 1)[0].strip()
        body = block
        sections.append((first_line[:120], body))
    return sections


def _merge_small_pieces(pieces: list[str]) -> list[str]:
    merged: list[str] = []
    buffer = ""
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if not buffer:
            buffer = piece
            continue
        if len(buffer) + len(piece) + 2 <= config.CHUNK_TARGET_CHARS:
            buffer = f"{buffer}\n\n{piece}"
        else:
            merged.append(buffer)
            buffer = piece
    if buffer:
        merged.append(buffer)
    return merged


def _split_oversized(text: str) -> list[str]:
    if len(text) <= config.CHUNK_MAX_CHARS:
        return [text]
    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + config.CHUNK_TARGET_CHARS)
        if end < len(text):
            break_at = text.rfind("\n", start, end)
            if break_at > start + config.CHUNK_TARGET_CHARS // 2:
                end = break_at
        parts.append(text[start:end].strip())
        start = max(end - config.CHUNK_OVERLAP_CHARS, start + 1)
    return [p for p in parts if p]


def _header(
    filename: str,
    page: int,
    section: str,
    company: str,
    ref_no: str,
) -> str:
    parts = [f"Doc: {filename}", f"Page {page}"]
    if section:
        parts.append(f"Section: {section}")
    if company:
        parts.append(f"Company: {company}")
    if ref_no:
        parts.append(f"Ref: {ref_no}")
    return "[" + " | ".join(parts) + "]"


def chunk_document(
    doc_id: str,
    filename: str,
    pages: list[tuple[int, str, str]],
    doc_level_text: str,
) -> list[ChunkRecord]:
    """
    pages: list of (page_num, text, source_type)
    """
    company = extract_company(doc_level_text)
    ref_no = extract_ref_no(doc_level_text)
    records: list[ChunkRecord] = []

    for page_num, page_text, source_type in pages:
        if not page_text.strip():
            continue
        sections = _split_sections(page_text)
        if not sections:
            sections = [("", page_text)]
        final_pieces: list[tuple[str, str]] = []
        for section_title, body in sections:
            for sub in _split_oversized(body):
                final_pieces.append((section_title, sub))

        seen: set[str] = set()
        for section, body in final_pieces:
            if not body.strip() or body in seen:
                continue
            seen.add(body)
            header = _header(filename, page_num, section, company, ref_no)
            full_text = f"{header}\n{body}"
            records.append(
                ChunkRecord(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    filename=filename,
                    page=page_num,
                    section=section[:120],
                    text=full_text,
                    source_type=source_type,
                    company=company,
                    ref_no=ref_no,
                )
            )
    return records
