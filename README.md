# Document RAG Chat

Upload PDFs, Excel/CSV, and images, index them locally (embeddings + BM25), and chat with **Gemini** using retrieved document chunks only.

## Features

- Digital PDF text extraction (watermark-aware) + **Tesseract OCR** fallback for scanned pages
- Excel/CSV and image uploads
- Local embeddings (`fastembed` / BGE-small) + **ChromaDB**
- Hybrid retrieval (vector + BM25, RRF) with optional cross-encoder reranking
- Streamlit chat with **source citations**
- Dockerized for easy sharing

## Quick start (local)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env and set GEMINI_API_KEY

streamlit run app.py
```

Open http://localhost:8501

**System dependency:** Tesseract must be installed on the host when running outside Docker.

- macOS: `brew install tesseract`
- Ubuntu: `sudo apt install tesseract-ocr tesseract-ocr-eng`

## Docker (recommended for sharing)

```bash
cp .env.example .env
# Add your GEMINI_API_KEY

docker compose up --build
```

Open http://localhost:8501

Indexed data persists in `./data` (mounted volume).

### Share with a friend

1. Send the repo (or a zip) plus instructions to create their own `.env` with a free Gemini key, **or**
2. Push a multi-arch image to Docker Hub and have them run:

   ```bash
   docker run -p 8501:8501 --env-file .env -v rag-data:/app/data your-image:tag
   ```

Never commit or bake `GEMINI_API_KEY` into the image.

## Bulk index sample docs (optional)

From the project root, you can index everything under `docs/` via a small script or the Streamlit uploader. For a one-off CLI ingest:

```bash
python - <<'PY'
from pathlib import Path
from rag.store import DocumentStore
from rag.ingest.loader import ingest_file

store = DocumentStore()
for path in sorted(Path("docs").glob("*")):
    if path.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg", ".xlsx", ".csv"}:
        r = ingest_file(path, store)
        print(path.name, "skipped" if r.skipped else f"{r.chunk_count} chunks")
PY
```

## Configuration

See `config.py` and environment variables in `.env.example` (`RAG_USE_RERANKER`, chunk sizes, OCR settings, etc.).

## Evaluation

Sample test questions live in `eval/questions.jsonl`. Use them to sanity-check retrieval on the fictional permit corpus.

## License

Sample documents in `docs/` are marked fictional for testing.
