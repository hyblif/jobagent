"""Tests for src/tools/web_search.py — all network calls mocked."""
import pytest

from src.tools.web_search import web_search


class FakeTavilyClient:
    def __init__(self, results):
        self._results = results

    def search(self, query, search_depth, max_results):
        return {"results": self._results}


class FailingTavilyClient:
    def __init__(self, exc):
        self._exc = exc

    def search(self, *args, **kwargs):
        raise self._exc


def _patch_client(monkeypatch, client):
    monkeypatch.setenv("TAVILY_API_KEY", "fake-key")
    monkeypatch.setattr("src.tools.web_search._client", lambda: client)


def test_basic_normalization(monkeypatch):
    fake_results = [
        {"title": "Title A", "url": "https://example.com/a", "content": "Content A" * 10, "score": 0.9},
        {"title": "Title B", "url": "https://example.com/b", "content": "Content B" * 10, "score": 0.8},
    ]
    _patch_client(monkeypatch, FakeTavilyClient(fake_results))
    results = web_search("test query")
    assert len(results) == 2
    assert results[0]["source_type"] == "web"
    assert results[0]["title"] == "Title A"
    assert results[0]["url_or_path"] == "https://example.com/a"
    assert results[0]["score"] == 0.9


def test_content_truncated_at_1200(monkeypatch):
    long_content = "x" * 2000
    fake_results = [{"title": "T", "url": "https://a.com", "content": long_content, "score": 1.0}]
    _patch_client(monkeypatch, FakeTavilyClient(fake_results))
    results = web_search("q")
    assert len(results[0]["excerpt"]) <= 1200


def test_no_api_key_returns_empty(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    results = web_search("anything")
    assert results == []


def test_exception_returns_empty(monkeypatch):
    _patch_client(monkeypatch, FailingTavilyClient(RuntimeError("network error")))
    # monkeypatch sleep to speed up retry backoff
    monkeypatch.setattr("time.sleep", lambda _: None)
    results = web_search("anything", retries=1)
    assert results == []


def test_missing_fields_handled(monkeypatch):
    fake_results = [{"url": "https://example.com"}]  # no title, content, score
    _patch_client(monkeypatch, FakeTavilyClient(fake_results))
    results = web_search("q")
    assert len(results) == 1
    assert results[0]["title"] == "(untitled)"
    assert results[0]["excerpt"] == ""
    assert results[0]["score"] is None
