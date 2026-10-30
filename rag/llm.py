"""Gemini client: rewrite, grounded generation, streaming."""

from __future__ import annotations

import os
from typing import Generator, Iterable

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

import config
from rag.store import RetrievedChunk

_client = None


def _get_client():
    global _client
    if _client is None:
        from google import genai

        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        _client = genai.Client(api_key=api_key)
    return _client


def _is_rate_limit(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "429" in msg or "rate" in msg or "quota" in msg or "resource_exhausted" in msg


@retry(
    retry=retry_if_exception(_is_rate_limit),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _generate_text(model: str, prompt: str) -> str:
    client = _get_client()
    response = client.models.generate_content(model=model, contents=prompt)
    return (response.text or "").strip()


def rewrite_follow_up(question: str, chat_history: list[dict]) -> str:
    if not chat_history:
        return question
    turns = chat_history[-config.REWRITE_HISTORY_TURNS * 2 :]
    history_lines = []
    for msg in turns:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        history_lines.append(f"{role.upper()}: {content}")
    history_block = "\n".join(history_lines)
    prompt = f"""Rewrite the latest user question into a standalone search query.
Use the conversation only for context. Output ONLY the rewritten query, nothing else.

Conversation:
{history_block}

Latest user question: {question}

Standalone query:"""
    try:
        rewritten = _generate_text(config.GEMINI_LITE_MODEL, prompt)
        return rewritten or question
    except Exception:
        return question


def build_rag_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return f"""You are a document assistant. No context was retrieved.

Question: {question}

Reply: I couldn't find that in the uploaded documents."""

    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        context_blocks.append(
            f"[{i}] ({chunk.filename}, p.{chunk.page}, {chunk.source_type})\n{chunk.text}"
        )
    context = "\n\n".join(context_blocks)
    return f"""You answer questions using ONLY the context below. Each block is labelled [n] with its source.
- If the answer is not in the context, say "I couldn't find that in the uploaded documents."
- Cite sources inline like [1], [3]. Quote numbers, dates and limits exactly as written.
- If documents conflict, say so and cite both.

Context:
{context}

Question: {question}
"""


@retry(
    retry=retry_if_exception(_is_rate_limit),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)
def stream_answer(prompt: str) -> Generator[str, None, None]:
    client = _get_client()
    stream = client.models.generate_content_stream(
        model=config.GEMINI_MODEL,
        contents=prompt,
    )
    for chunk in stream:
        text = getattr(chunk, "text", None)
        if text:
            yield text


def generate_answer(prompt: str) -> Iterable[str]:
    """Wrapper for Streamlit write_stream."""
    yield from stream_answer(prompt)
