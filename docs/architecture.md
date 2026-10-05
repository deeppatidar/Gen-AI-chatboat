# Architecture

The UI is Next.js. All RAG work is Python (FastAPI). Next.js rewrites `/api/*` to `http://127.0.0.1:8000`.

The app never ships a sample PDF or policy text. Knowledge exists only after the user uploads a file.

## Ingestion (upload time)

1. Validate PDF/TXT from file bytes (not the filename).
2. SHA-256 duplicate check.
3. Extract text (`pypdf` keeps page numbers).
4. Split on paragraph/sentence boundaries with `CHUNK_SIZE_WORDS=1000` and `CHUNK_OVERLAP_WORDS=150`.
5. Embed each chunk (OpenAI `text-embedding-3-small`).
6. Store chunk text + embedding + metadata in persistent ChromaDB (`data/chroma`).

## Question time

1. Embed the question only.
2. Cosine search, `TOP_K=5`, drop hits below `MIN_RELEVANCE_SCORE` (default `0.18` for OpenAI embeddings).
3. If nothing is relevant, return a not-found message (no LLM guess).
4. Otherwise send the retrieved chunks to the LLM and return the answer plus sources.

Documents are not re-read, re-chunked, or re-embedded on each question.

See [workflow.md](./workflow.md) for full flowcharts (ingestion, question, delete) and step-by-step tables.

## Layout

- `backend/rag.py` — extract, chunk, embed, store, retrieve, LLM
- `backend/app.py` — HTTP routes
- `components/ChatApp.tsx` — upload + chat UI
