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
    retries: int = 2,
) -> list[dict]:
    """Search the web and return raw evidence dicts (id assigned later in rerank_node).

    Returns list of dicts: {source_type, title, url_or_path, excerpt, score}.
    Always returns a list; never raises.
    """
    client = _client()
    if client is None:
        return []

    for attempt in range(retries + 1):
        try:
            resp = client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results,
            )
            break
        except Exception:
            if attempt < retries:
                time.sleep(2**attempt)
            else:
                return []

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
