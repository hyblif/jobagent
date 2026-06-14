import os
import time


def _client():
    from tavily import TavilyClient

    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        return None
    return TavilyClient(api_key=key)


def web_search(
    query: str,
    max_results: int = 5,
    retries: int = 1,
    timeout: int = 15,
) -> list[dict]:
    """Search the web and return raw evidence dicts (id assigned later in rerank_node).

    Returns list of dicts: {source_type, title, url_or_path, excerpt, score}.
    Raises on missing configuration or final search failure so workflow nodes can warn.
    """
    client = _client()
    if client is None:
        raise RuntimeError("TAVILY_API_KEY 未配置")

    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results,
                timeout=timeout,
            )
            break
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(2**attempt)
            else:
                raise RuntimeError(f"Tavily 搜索失败：{exc}") from exc
    else:
        raise RuntimeError(f"Tavily 搜索失败：{last_exc}")

    out = []
    for r in resp.get("results", []):
        out.append(
            {
                "source_type": "web",
                "title": r.get("title") or "(untitled)",
                "url_or_path": r.get("url", ""),
                "excerpt": (r.get("content") or "")[:1200],
                "score": r.get("score"),
            }
        )
    return out
