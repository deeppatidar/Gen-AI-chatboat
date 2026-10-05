from __future__ import annotations

import hashlib
import io
import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
load_dotenv(ROOT / ".env.local", override=True)

WORD_RE = re.compile(r"\S+")
LOCK = threading.Lock()
_COLLECTION = None
_COLLECTION_DIR: str | None = None

GROUNDED_SYSTEM_PROMPT = """You are a document-grounded assistant.
Answer using only the retrieved knowledge below.
If it is not enough, say: I couldn't find information about this in the uploaded documents.
Do not invent facts. Do not use outside knowledge. Cite sources with [n]."""

NO_CONTEXT = "No sufficiently relevant information was found in the uploaded documents."
NOT_FOUND = "I couldn't find information about this in the uploaded documents."
LLM_DOWN = "Unable to generate a response right now. Please try again."


class AppError(Exception):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw else default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw else default


def data_dir() -> Path:
    return Path(os.getenv("DATA_DIR", ROOT / "data"))


def documents_path() -> Path:
    return data_dir() / "documents.json"


def chunk_size() -> int:
    return _env_int("CHUNK_SIZE_WORDS", 1000)


def chunk_overlap() -> int:
    return _env_int("CHUNK_OVERLAP_WORDS", 150)


def top_k() -> int:
    return _env_int("TOP_K", 5)


def min_score() -> float:
    # text-embedding-3-small cosine similarity for on-topic questions is often 0.20–0.50.
    # 0.35 was rejecting real matches from uploaded documents.
    return _env_float("MIN_RELEVANCE_SCORE", 0.18)


def max_file_bytes() -> int:
    return _env_int("MAX_FILE_SIZE_MB", 10) * 1024 * 1024


def max_history() -> int:
    return _env_int("MAX_HISTORY_MESSAGES", 8)


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def clean_text(text: str) -> str:
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def detect_type(content: bytes, filename: str, mime: str) -> str:
    if content.startswith(b"%PDF"):
        return "pdf"
    ext = Path(filename).suffix.lower()
    mime = (mime or "").lower()
    if ext == ".pdf" or mime == "application/pdf":
        raise AppError(400, "Unable to process the document.")
    if ext == ".txt" or mime == "text/plain":
        try:
            content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise AppError(400, "Unable to process the document.") from exc
        return "txt"
    raise AppError(400, "Only PDF and TXT files are supported.")


def extract_pages(content: bytes, source_type: str) -> list[dict[str, Any]]:
    if source_type == "txt":
        return [{"page_number": None, "text": clean_text(content.decode("utf-8"))}]
    try:
        reader = PdfReader(io.BytesIO(content))
        pages = []
        for index, page in enumerate(reader.pages, start=1):
            pages.append(
                {
                    "page_number": index,
                    "text": clean_text(page.extract_text() or ""),
                }
            )
        return pages
    except Exception as exc:
        raise AppError(400, "Unable to process the document.") from exc


def _split_oversized(text: str, max_words: int) -> list[str]:
    separators = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]

    def recurse(value: str, sep_index: int) -> list[str]:
        if word_count(value) <= max_words:
            return [value.strip()] if value.strip() else []
        if sep_index >= len(separators):
            words = WORD_RE.findall(value)
            return [
                " ".join(words[i : i + max_words])
                for i in range(0, len(words), max_words)
            ]
        delimiter = separators[sep_index]
        parts = value.split(delimiter)
        if len(parts) == 1:
            return recurse(value, sep_index + 1)
        chunks: list[str] = []
        current = ""
        for i, part in enumerate(parts):
            piece = part if i == len(parts) - 1 else part + delimiter
            candidate = current + piece
            if word_count(candidate) <= max_words:
                current = candidate
                continue
            if current.strip():
                chunks.append(current.strip())
            if word_count(piece) > max_words:
                chunks.extend(recurse(piece, sep_index + 1))
                current = ""
            else:
                current = piece
        if current.strip():
            chunks.append(current.strip())
        return chunks

    return recurse(text, 0)


def chunk_text(text: str, size: int | None = None, overlap: int | None = None) -> list[str]:
    size = size or chunk_size()
    overlap = overlap if overlap is not None else chunk_overlap()
    cleaned = clean_text(text)
    if not cleaned:
        return []
    pieces = _split_oversized(cleaned, size)
    if len(pieces) <= 1 or overlap <= 0:
        return pieces
    overlapped = [pieces[0]]
    for i in range(1, len(pieces)):
        previous = WORD_RE.findall(pieces[i - 1])[-overlap:]
        prefix = " ".join(previous)
        nxt = pieces[i]
        overlapped.append(f"{prefix} {nxt}".strip() if prefix and not nxt.startswith(prefix) else nxt)
    return overlapped


def chunk_pages(
    pages: list[dict[str, Any]],
    document_id: str,
    document_name: str,
    source_type: str,
    file_hash: str,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    index = 0
    for page in pages:
        for text in chunk_text(page["text"]):
            chunks.append(
                {
                    "chunkId": f"{document_id}_chunk_{index}",
                    "documentId": document_id,
                    "documentName": document_name,
                    "text": text,
                    "pageNumber": page["page_number"],
                    "chunkIndex": index,
                    "sourceType": source_type,
                    "fileHash": file_hash,
                }
            )
            index += 1
    return chunks


class HashEmbeddingFunction(EmbeddingFunction[Documents]):
    """Deterministic embeddings for tests only. The running app uses OpenAI."""

    def __init__(self, dimensions: int = 64):
        self.dimensions = dimensions

    def name(self) -> str:
        return "hash"

    def __call__(self, input: Documents) -> Embeddings:
        return [self._embed(text) for text in input]

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        words = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2]
        for word in words:
            bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dimensions
            vector[bucket] += 1.0
        for i in range(len(words) - 1):
            bigram = f"{words[i]}_{words[i + 1]}"
            bucket = int(hashlib.md5(bigram.encode()).hexdigest(), 16) % self.dimensions
            vector[bucket] += 2.0
        norm = sum(value * value for value in vector) ** 0.5
        if not norm:
            return vector
        return [value / norm for value in vector]


def _embedding_function() -> Any:
    if os.getenv("PYTEST_CURRENT_TEST") or os.getenv("EMBEDDING_PROVIDER") == "fake":
        return HashEmbeddingFunction()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise AppError(500, "The OpenAI API key is not configured.")
    from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

    kwargs: dict[str, Any] = {
        "api_key": api_key,
        "model_name": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    }
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        kwargs["api_base"] = base_url
    return OpenAIEmbeddingFunction(**kwargs)


def reset_vector_store() -> None:
    global _COLLECTION, _COLLECTION_DIR
    _COLLECTION = None
    _COLLECTION_DIR = None


def get_collection():
    global _COLLECTION, _COLLECTION_DIR
    directory = str(data_dir() / "chroma")
    if _COLLECTION is not None and _COLLECTION_DIR == directory:
        return _COLLECTION
    Path(directory).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=directory)
    _COLLECTION = client.get_or_create_collection(
        name="document_chunks",
        embedding_function=_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )
    _COLLECTION_DIR = directory
    return _COLLECTION


def _read_documents() -> list[dict[str, Any]]:
    path = documents_path()
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("documents", [])


def _write_documents(documents: list[dict[str, Any]]) -> None:
    path = documents_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"documents": documents}, indent=2), encoding="utf-8")


def list_documents() -> list[dict[str, Any]]:
    docs = _read_documents()
    return sorted(docs, key=lambda item: item.get("uploadedAt", ""), reverse=True)


def indexed_count() -> int:
    return sum(1 for doc in _read_documents() if doc.get("status") == "indexed")


def upsert_chunks(chunks: list[dict[str, Any]]) -> None:
    if not chunks:
        return
    collection = get_collection()
    collection.add(
        ids=[chunk["chunkId"] for chunk in chunks],
        documents=[chunk["text"] for chunk in chunks],
        metadatas=[
            {
                "documentId": chunk["documentId"],
                "documentName": chunk["documentName"],
                "pageNumber": chunk["pageNumber"] or 0,
                "chunkIndex": chunk["chunkIndex"],
                "sourceType": chunk["sourceType"],
                "fileHash": chunk["fileHash"],
            }
            for chunk in chunks
        ],
    )


def delete_chunks(document_id: str) -> None:
    collection = get_collection()
    existing = collection.get(where={"documentId": document_id})
    ids = existing.get("ids") or []
    if ids:
        collection.delete(ids=ids)


def count_chunks(document_id: str) -> int:
    collection = get_collection()
    existing = collection.get(where={"documentId": document_id})
    return len(existing.get("ids") or [])


def search_chunks(question: str, k: int | None = None, threshold: float | None = None) -> list[dict[str, Any]]:
    collection = get_collection()
    if collection.count() == 0:
        return []
    k = k or top_k()
    threshold = min_score() if threshold is None else threshold
    result = collection.query(
        query_texts=[question],
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    hits: list[dict[str, Any]] = []
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    ids = (result.get("ids") or [[]])[0]
    for chunk_id, text, meta, distance in zip(ids, documents, metadatas, distances):
        score = max(0.0, min(1.0, 1.0 - float(distance)))
        page = int(meta.get("pageNumber") or 0)
        hits.append(
            {
                "chunkId": chunk_id,
                "text": text,
                "documentId": meta.get("documentId"),
                "documentName": meta.get("documentName"),
                "pageNumber": page if page > 0 else None,
                "chunkIndex": meta.get("chunkIndex"),
                "sourceType": meta.get("sourceType"),
                "relevanceScore": round(score, 4),
            }
        )
    hits.sort(key=lambda item: item["relevanceScore"], reverse=True)
    return [hit for hit in hits if hit["relevanceScore"] >= threshold]


def ingest_file(content: bytes, filename: str, mime: str, size: int) -> dict[str, Any]:
    if size <= 0:
        raise AppError(400, "The uploaded document does not contain usable text.")
    if size > max_file_bytes():
        raise AppError(400, f"File is too large. Maximum size is {max_file_bytes() // (1024 * 1024)} MB.")

    source_type = detect_type(content, filename, mime)
    name = Path(filename.replace("\\", "/")).name or "document"
    file_hash = sha256_bytes(content)

    with LOCK:
        documents = _read_documents()
        existing = next((doc for doc in documents if doc.get("fileHash") == file_hash), None)
        if existing and existing.get("status") == "indexed":
            return {
                "documentId": existing["id"],
                "name": existing["name"],
                "status": existing["status"],
                "chunkCount": existing.get("chunkCount", 0),
                "duplicate": True,
                "message": "This document has already been indexed.",
            }

        document_id = existing["id"] if existing else str(uuid.uuid4())
        record = {
            "id": document_id,
            "name": name,
            "type": source_type,
            "size": size,
            "fileHash": file_hash,
            "uploadedAt": existing.get("uploadedAt") if existing else datetime.now(timezone.utc).isoformat(),
            "status": "processing",
            "chunkCount": 0,
            "error": None,
        }
        documents = [doc for doc in documents if doc["id"] != document_id]
        documents.append(record)
        _write_documents(documents)

    try:
        pages = extract_pages(content, source_type)
        full_text = "\n\n".join(page["text"] for page in pages).strip()
        if not full_text:
            raise AppError(400, "The uploaded document does not contain usable text.")
        chunks = chunk_pages(pages, document_id, name, source_type, file_hash)
        if not chunks:
            raise AppError(400, "The uploaded document does not contain usable text.")
        delete_chunks(document_id)
        upsert_chunks(chunks)
        with LOCK:
            documents = _read_documents()
            for doc in documents:
                if doc["id"] == document_id:
                    doc["status"] = "indexed"
                    doc["chunkCount"] = len(chunks)
                    doc["error"] = None
            _write_documents(documents)
        return {
            "documentId": document_id,
            "name": name,
            "status": "indexed",
            "chunkCount": len(chunks),
        }
    except Exception as exc:
        message = exc.message if isinstance(exc, AppError) else "Unable to process the document."
        with LOCK:
            documents = _read_documents()
            for doc in documents:
                if doc["id"] == document_id:
                    doc["status"] = "failed"
                    doc["error"] = message
            _write_documents(documents)
        if isinstance(exc, AppError):
            raise
        raise AppError(500, message) from exc


def remove_document(document_id: str) -> None:
    with LOCK:
        documents = _read_documents()
        if not any(doc["id"] == document_id for doc in documents):
            raise AppError(404, "Document not found.")
        _write_documents([doc for doc in documents if doc["id"] != document_id])
    delete_chunks(document_id)


def _openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise AppError(503, LLM_DOWN)
    return OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_BASE_URL") or None)


def _complete(system: str, user: str, history: list[dict[str, str]]) -> str:
    if os.getenv("PYTEST_CURRENT_TEST") or os.getenv("EMBEDDING_PROVIDER") == "fake":
        return f"According to the retrieved documents: {user[:280]}"
    try:
        response = _openai_client().chat.completions.create(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                *history,
                {"role": "user", "content": user},
            ],
        )
        content = (response.choices[0].message.content or "").strip()
        if not content:
            raise AppError(503, LLM_DOWN)
        return content
    except AppError:
        raise
    except Exception as exc:
        raise AppError(503, LLM_DOWN) from exc


def _excerpt(text: str, limit: int = 280) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    return cleaned if len(cleaned) <= limit else cleaned[:limit].rstrip() + "…"


def answer_question(message: str, history: list[dict[str, str]] | None = None) -> dict[str, Any]:
    question = (message or "").strip()
    if not question:
        raise AppError(400, "Please enter a question.")
    if indexed_count() == 0:
        raise AppError(400, "Upload a PDF or TXT document before asking a question.")

    safe_history = []
    for item in (history or [])[-max_history():]:
        role = item.get("role")
        content = (item.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            safe_history.append({"role": role, "content": content[:4000]})

    chunks = search_chunks(question)
    if not chunks:
        return {"answer": NO_CONTEXT, "sources": [], "mode": "rag"}

    context = "\n\n".join(
        f"[{i}] Document: {chunk['documentName']}"
        + (f", Page: {chunk['pageNumber']}" if chunk.get("pageNumber") else "")
        + f"\n{chunk['text']}"
        for i, chunk in enumerate(chunks, start=1)
    )
    answer = _complete(
        GROUNDED_SYSTEM_PROMPT,
        f"Retrieved knowledge:\n{context}\n\nUser question:\n{question}",
        safe_history,
    )
    missing = "couldn't find information" in answer.lower() or "could not find information" in answer.lower()
    sources = [] if missing else [
        {
            "documentId": chunk["documentId"],
            "documentName": chunk["documentName"],
            "pageNumber": chunk.get("pageNumber"),
            "chunkId": chunk["chunkId"],
            "relevanceScore": chunk["relevanceScore"],
            "excerpt": _excerpt(chunk["text"]),
        }
        for chunk in chunks
    ]
    return {
        "answer": NOT_FOUND if missing else answer,
        "sources": sources,
        "mode": "rag",
    }
