from fastapi.testclient import TestClient

from app import app
from rag import chunk_text, ingest_file, search_chunks


client = TestClient(app)


def test_chunking_uses_overlap():
    words = " ".join(f"word{i}" for i in range(250))
    chunks = chunk_text(words, size=100, overlap=20)
    assert len(chunks) > 1
    tail = " ".join(chunks[0].split()[-20:])
    assert chunks[1].startswith(tail)


def test_txt_upload_chunks_and_embeds():
    text = (
        "Employees may carry forward up to 15 days of unused annual leave. "
        "Sick leave requires a medical certificate after three days."
    )
    result = ingest_file(text.encode(), "leave-policy.txt", "text/plain", len(text))
    assert result["status"] == "indexed"
    assert result["chunkCount"] >= 1

    hits = search_chunks("How many annual leave days can employees carry forward?", threshold=0.05)
    assert hits
    assert hits[0]["documentName"] == "leave-policy.txt"


def test_duplicate_upload_is_detected():
    body = b"Unique security policy about multi factor authentication."
    first = ingest_file(body, "security.txt", "text/plain", len(body))
    second = ingest_file(body, "security.txt", "text/plain", len(body))
    assert first["status"] == "indexed"
    assert second["duplicate"] is True


def test_unsupported_file_rejected():
    response = client.post(
        "/api/documents/upload",
        files={"file": ("photo.png", b"\x89PNG", "image/png")},
    )
    assert response.status_code == 400
    assert "Only PDF and TXT" in response.json()["error"]


def test_empty_file_rejected():
    response = client.post(
        "/api/documents/upload",
        files={"file": ("empty.txt", b"   \n", "text/plain")},
    )
    assert response.status_code == 400


def test_chat_requires_upload_and_then_uses_retrieval():
    empty = client.post("/api/chat", json={"message": "What is the leave policy?"})
    assert empty.status_code == 400
    assert "Upload" in empty.json()["error"]

    text = "Employees may carry forward up to 15 days of unused annual leave."
    uploaded = client.post(
        "/api/documents/upload",
        files={"file": ("leave-policy.txt", text.encode(), "text/plain")},
    )
    assert uploaded.status_code == 200

    listed = client.get("/api/documents")
    assert listed.status_code == 200
    assert listed.json()["documents"][0]["status"] == "indexed"

    chat = client.post(
        "/api/chat",
        json={"message": "How many annual leave days can employees carry forward?"},
    )
    assert chat.status_code == 200
    body = chat.json()
    assert body["sources"][0]["documentName"] == "leave-policy.txt"
    assert body["answer"]

    missing = client.post(
        "/api/chat",
        json={"message": "What is the policy for colonizing Mars with plutonium rockets?"},
    )
    assert missing.status_code == 200
    assert "uploaded documents" in missing.json()["answer"].lower()

    blank = client.post("/api/chat", json={"message": "   "})
    assert blank.status_code == 400

    document_id = uploaded.json()["documentId"]
    deleted = client.delete(f"/api/documents/{document_id}")
    assert deleted.status_code == 200
    gone = client.delete(f"/api/documents/{document_id}")
    assert gone.status_code == 404
