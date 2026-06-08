import os
from functools import lru_cache


@lru_cache(maxsize=1)
def _load_reranker():
    from FlagEmbedding import FlagReranker

    model_path = os.environ.get("RERANKER_MODEL_PATH") or os.environ.get(
        "RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3"
    )
    use_fp16 = os.environ.get("RERANKER_FP16", "false").lower() == "true"
    return FlagReranker(model_path, use_fp16=use_fp16)


def rerank(
    query: str,
    candidates: list[dict],
    text_key: str = "excerpt",
    top_k: int = 5,
) -> list[dict]:
    if not candidates:
        return []

    reranker = _load_reranker()
    pairs = [[query, c[text_key]] for c in candidates]
    scores = reranker.compute_score(pairs, normalize=True)

    if not isinstance(scores, list):
        scores = [scores]

    # annotate in place then sort
    candidates = [dict(c) for c in candidates]
    for c, s in zip(candidates, scores):
        c["score"] = float(s)

    return sorted(candidates, key=lambda c: c["score"], reverse=True)[:top_k]
