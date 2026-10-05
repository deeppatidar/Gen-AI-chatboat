# Workflow and Flowcharts

This document describes how the RAG Assistant moves data from an uploaded file to a grounded answer.

**Stack**

- UI: Next.js (`components/ChatApp.tsx`) at http://localhost:3000
- API: Python FastAPI (`backend/app.py`) at http://localhost:8000
- Pipeline: `backend/rag.py`
- Next.js rewrites `/api/*` to the Python server

Knowledge exists only after a user uploads a PDF or TXT. Nothing is hardcoded.

---

## End-to-end picture

```mermaid
flowchart TB
  subgraph ui [Browser - Next.js]
    U[User]
    Chat[Chat UI]
  end

  subgraph api [Python FastAPI]
    UploadAPI["POST /api/documents/upload"]
    ChatAPI["POST /api/chat"]
    ListAPI["GET /api/documents"]
    DelAPI["DELETE /api/documents/:id"]
  end

  subgraph store [Disk - data/]
    JSON["documents.json<br/>catalog"]
    Chroma["ChromaDB<br/>chunks + embeddings"]
  end

  subgraph cloud [OpenAI]
    Emb[text-embedding-3-small]
    LLM[gpt-4o-mini]
  end

  U --> Chat
  Chat --> UploadAPI
  Chat --> ChatAPI
  Chat --> ListAPI
  Chat --> DelAPI

  UploadAPI --> JSON
  UploadAPI --> Emb
  UploadAPI --> Chroma

  ChatAPI --> Emb
  ChatAPI --> Chroma
  ChatAPI --> LLM
```

Two rules:

1. **Upload time** is when the document is extracted, chunked, embedded, and stored.
2. **Question time** only embeds the question, searches stored vectors, and calls the LLM. The original file is not read again.

---

## Workflow A — Document ingestion

Triggered by **+ Upload** in the sidebar. Route: `POST /api/documents/upload` → `ingest_file()`.

```mermaid
flowchart TD
  A[User selects PDF or TXT] --> B[Browser sends multipart file]
  B --> C[Python reads file bytes]
  C --> D{Size valid and type PDF or TXT?}
  D -->|No| E[400 error<br/>unsupported / empty / too large]
  D -->|Yes| F[SHA-256 hash of bytes]
  F --> G{Same hash already indexed?}
  G -->|Yes| H[409<br/>This document has already been indexed]
  G -->|No| I[Write status processing to documents.json]
  I --> J[Extract text]
  J --> K{PDF?}
  K -->|Yes| L[pypdf: text per page + page number]
  K -->|No| M[UTF-8 text, no page number]
  L --> N[Clean whitespace]
  M --> N
  N --> O{Usable text?}
  O -->|No| P[Mark failed in JSON]
  O -->|Yes| Q[Chunk: 1000 words, 150 overlap<br/>prefer paragraph then sentence boundaries]
  Q --> R[Embed each chunk<br/>OpenAI text-embedding-3-small]
  R --> S[Store in Chroma collection document_chunks<br/>text + vector + metadata]
  S --> T[Update documents.json<br/>status indexed, chunkCount]
  T --> U[UI shows Indexed]
```

### Step by step

| Step | What happens | Code |
| --- | --- | --- |
| 1 | File is sent from the browser | `ChatApp` → `POST /api/documents/upload` |
| 2 | Type is detected from **bytes** (`%PDF` or UTF-8 `.txt`), not the filename | `detect_type()` |
| 3 | SHA-256 identifies exact duplicates | `sha256_bytes()` |
| 4 | Text is extracted. PDFs keep page numbers | `extract_pages()` |
| 5 | Text is split with overlap | `chunk_text()` / `chunk_pages()` |
| 6 | Each chunk is embedded (1536 numbers) | Chroma + OpenAI embedding function |
| 7 | Vectors and chunk text are written to disk | `data/chroma/` |
| 8 | Catalog row is updated | `data/documents.json` |

The original file is **not** saved. After this workflow, only chunks + embeddings + metadata remain.

### Chunking defaults

Configured in `.env` / `.env.example`:

- `CHUNK_SIZE_WORDS=1000`
- `CHUNK_OVERLAP_WORDS=150`

Split order: paragraphs → newlines → sentences → commas/spaces → hard word window.

---

## Workflow B — Ask a question

Triggered by **Send**. Route: `POST /api/chat` → `answer_question()`.

```mermaid
flowchart TD
  A[User types a question] --> B{Any indexed documents?}
  B -->|No| C[400<br/>Upload a document first]
  B -->|Yes| D[Embed the question only]
  D --> E[Chroma cosine search<br/>top K = 5]
  E --> F[Convert distance to score<br/>score = 1 - distance]
  F --> G{Any hit with score ≥ 0.18?}
  G -->|No| H[Return not found<br/>do not call LLM]
  G -->|Yes| I[Build grounded prompt<br/>question + numbered excerpts]
  I --> J[LLM gpt-4o-mini<br/>answer only from retrieved text]
  J --> K[Return answer + sources<br/>document name, page, excerpt]
  K --> L[UI shows message and expandable sources]
```

### Step by step

| Step | What happens | Code |
| --- | --- | --- |
| 1 | Empty question is rejected in the UI and API | `answer_question()` |
| 2 | If no indexed docs, chat is blocked | `indexed_count()` |
| 3 | Only the **question** is embedded | `collection.query(query_texts=...)` |
| 4 | Top 5 nearest chunks across **all** indexed files | `TOP_K=5` |
| 5 | Hits below `MIN_RELEVANCE_SCORE` (0.18) are dropped | `search_chunks()` |
| 6 | If none remain: *No sufficiently relevant information was found...* | no LLM call |
| 7 | Otherwise the LLM sees numbered excerpts and must cite `[n]` | `GROUNDED_SYSTEM_PROMPT` |
| 8 | UI lists sources (name, page for PDFs, short excerpt) | `ChatApp` |

Chat bubbles live in the **browser session**. They are not written to Chroma. Indexed documents **are** on disk and survive restart.

---

## Workflow C — List and delete

```mermaid
flowchart LR
  subgraph list [List]
    G[GET /api/documents] --> J[Read documents.json]
    J --> S[Sidebar]
  end

  subgraph del [Delete]
    D[DELETE /api/documents/id] --> V[Remove all Chroma vectors for that id]
    V --> M[Remove row from documents.json]
  end
```

Delete removes both catalog and vectors so nothing is left behind.

---

## Where data lives

```text
data/
  documents.json      ← human-readable catalog (sidebar)
  chroma/
    chroma.sqlite3    ← Chroma catalog + embeddings metadata
    <uuid>/           ← vector segment files
```

| Store | Survives refresh / restart | What you can open |
| --- | --- | --- |
| `documents.json` | Yes | Directly in the editor |
| ChromaDB `document_chunks` | Yes | Query with Python or open `chroma.sqlite3` |
| Original PDF/TXT | No | Not kept after ingest |
| Chat history | No | Browser memory only |

Inspect stored chunks:

```bash
cd backend
../.venv/bin/python - <<'PY'
from rag import get_collection
col = get_collection()
got = col.get(include=["documents", "metadatas"])
print("chunks:", col.count())
for cid, text, meta in zip(got["ids"], got["documents"], got["metadatas"]):
    print(meta["documentName"], "page", meta.get("pageNumber"), "chunk", meta.get("chunkIndex"))
    print((text or "")[:200], "\n")
PY
```

---

## Combined flowchart

```text
                    ┌─────────────────┐
                    │   User Upload   │
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │ Text Extraction │
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │    Chunking     │
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │   Embeddings    │
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │  Vector DB      │
                    │  data/chroma    │
                    └─────────────────┘


User Question
      ▼
Query Embedding
      ▼
Vector Search
      ▼
Relevant Chunks  (score ≥ 0.18)
      ▼
LLM
      ▼
Grounded Answer + Sources
```

---

## API map

| Method | Path | Workflow |
| --- | --- | --- |
| `POST` | `/api/documents/upload` | Ingestion |
| `GET` | `/api/documents` | List catalog |
| `DELETE` | `/api/documents/:id` | Delete catalog + vectors |
| `POST` | `/api/chat` | Retrieve + generate |

Related: [architecture.md](./architecture.md)
