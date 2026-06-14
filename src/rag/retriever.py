from src.rag.rerank import rerank
from src.rag.store import get_collection


class KnowledgeBaseEmptyError(RuntimeError):
    """Raised when the vector store exists but has no indexed chunks."""


def retrieve(
    query: str,
    n_candidates: int = 20,
    top_k: int = 5,
    use_rerank: bool = True,
    raise_on_error: bool = False,
) -> list[dict]:
    """Returns candidate dicts: {id, source_type, title, url_or_path, excerpt, score}"""
    try:
        collection = get_collection()
    except Exception as exc:
        if raise_on_error:
            raise RuntimeError(f"知识库连接失败：{exc}") from exc
        return []

    if collection.count() == 0:
        if raise_on_error:
            raise KnowledgeBaseEmptyError("知识库为空，建议先 build_index")
        return []

    n_results = min(n_candidates, collection.count())
    try:
        res = collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        if raise_on_error:
            raise RuntimeError(f"知识库查询失败：{exc}") from exc
        return []

    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    ids = res["ids"][0]

    candidates = []
    for cid, doc, meta, dist in zip(ids, docs, metas, dists):
        candidates.append(
            {
                "id": cid,
                "source_type": "kb",
                "title": meta.get("title") or meta.get("source", "knowledge base"),
                "url_or_path": meta.get("source", ""),
                "excerpt": doc,
                "score": 1.0 - float(dist),  # cosine sim fallback
            }
        )

    if not candidates:
        return []

    if use_rerank:
        return rerank(query, candidates, text_key="excerpt", top_k=top_k)

    return sorted(candidates, key=lambda c: c["score"], reverse=True)[:top_k]
