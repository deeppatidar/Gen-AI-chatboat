# RAG Assistant

Next.js chat UI + Python FastAPI backend. There is **no built-in PDF or text**. You upload a document; the backend chunks it, embeds it, and stores the vectors. Questions search those stored chunks and the LLM answers only from that retrieved context.

```text
Upload PDF/TXT → extract → chunk → embed → ChromaDB
Question → embed query → vector search → LLM → answer + sources
```

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
cp .env.example .env.local
# set OPENAI_API_KEY in .env.local

# terminal 1
npm run backend

# terminal 2
npm install
npm run dev
```

Open http://localhost:3000, upload a PDF or TXT, wait until it is **Indexed**, then ask a question.

## Tests

```bash
source .venv/bin/activate
npm test
```

Tests generate temporary text in memory. They do not load a project document into the app.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/api/documents/upload` | Extract, chunk, embed, store |
| GET | `/api/documents` | List uploaded documents |
| DELETE | `/api/documents/:id` | Remove metadata + vectors |
| POST | `/api/chat` | Retrieve chunks + grounded LLM answer |

More detail:

- [Workflow and flowcharts](docs/workflow.md)
- [Architecture](docs/architecture.md)
# Gen-AI-chatboat
