from __future__ import annotations

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from rag import AppError, answer_question, ingest_file, list_documents, remove_document

app = FastAPI(title="RAG Assistant")


class ChatRequest(BaseModel):
    message: str = ""
    history: list[dict] = []


@app.exception_handler(AppError)
async def app_error_handler(_request, exc: AppError):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.message})


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/documents")
def get_documents():
    return {"documents": list_documents()}


@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    content = await file.read()
    result = ingest_file(
        content=content,
        filename=file.filename or "document",
        mime=file.content_type or "",
        size=len(content),
    )
    status = 409 if result.get("duplicate") else 200
    return JSONResponse(status_code=status, content=result)


@app.delete("/api/documents/{document_id}")
def delete_document(document_id: str):
    remove_document(document_id)
    return {"ok": True}


@app.post("/api/chat")
def chat(body: ChatRequest):
    return answer_question(body.message, body.history)
