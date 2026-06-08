"""pytest fixtures: isolate Chroma to tmp_path and clear lru_caches between tests."""
import os

import pytest


@pytest.fixture(autouse=True)
def isolate_chroma(tmp_path, monkeypatch):
    """Each test gets its own Chroma directory."""
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    # Clear cached client so it picks up the new path
    from src.rag import store
    store.get_client.cache_clear()
    yield
    store.get_client.cache_clear()


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Guard: unset real API keys so tests never accidentally hit the network."""
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
