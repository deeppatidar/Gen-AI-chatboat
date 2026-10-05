import pytest

from rag import reset_vector_store


@pytest.fixture(autouse=True)
def isolated_data(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fake")
    monkeypatch.setenv("MIN_RELEVANCE_SCORE", "0.2")
    reset_vector_store()
    yield
    reset_vector_store()
