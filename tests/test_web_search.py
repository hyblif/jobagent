"""Tests for src/tools/web_search.py — all network calls mocked."""
import pytest

from src.tools.web_search import web_search


class FakeTavilyClient:
    def __init__(self, results):
        self._results = results

    def search(self, query, search_depth, max_results, timeout):
        return {"results": self._results}


class FailingTavilyClient:
    def __init__(self, exc):
        self._exc = exc

    def search(self, *args, **kwargs):
        raise self._exc


class FlakyTavilyClient:
    def __init__(self, results):
        self._results = results
        self.calls = 0

    def search(self, *args, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise TimeoutError("temporary timeout")
        return {"results": self._results}


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


def test_no_api_key_raises(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="TAVILY_API_KEY"):
        web_search("anything")


def test_exception_raises_after_retries(monkeypatch):
    _patch_client(monkeypatch, FailingTavilyClient(RuntimeError("network error")))
    # monkeypatch sleep to speed up retry backoff
    monkeypatch.setattr("time.sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="Tavily 搜索失败"):
        web_search("anything", retries=1)


def test_retry_success_returns_normalized_results(monkeypatch):
    fake_results = [
        {"title": "Retry OK", "url": "https://example.com/retry", "content": "Recovered", "score": 0.7}
    ]
    client = FlakyTavilyClient(fake_results)
    _patch_client(monkeypatch, client)
    monkeypatch.setattr("time.sleep", lambda _: None)
    results = web_search("retry query", retries=1)
    assert client.calls == 2
    assert results[0]["title"] == "Retry OK"
    assert results[0]["url_or_path"] == "https://example.com/retry"


def test_missing_fields_handled(monkeypatch):
    fake_results = [{"url": "https://example.com"}]  # no title, content, score
    _patch_client(monkeypatch, FakeTavilyClient(fake_results))
    results = web_search("q")
    assert len(results) == 1
    assert results[0]["title"] == "(untitled)"
    assert results[0]["excerpt"] == ""
    assert results[0]["score"] is None
