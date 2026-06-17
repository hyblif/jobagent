from src.eval.retrieval_eval import BGE_QUERY_PREFIX, scan_parameters, score_results
from src.eval import retrieval_eval


GOLD = [{"source": "llm_rag.md", "heading": "RAG 流程"}]


def test_score_results_rank_one_hit():
    results = [
        {"source": "llm_rag.md", "heading": "RAG 流程"},
        {"source": "network.md", "heading": "HTTP 与 HTTPS"},
    ]

    score = score_results(results, GOLD)

    assert score["rank"] == 1
    assert score["hit@3"] is True
    assert score["hit@5"] is True
    assert score["mrr"] == 1.0


def test_score_results_rank_four_hit():
    results = [
        {"source": "network.md", "heading": "HTTP 与 HTTPS"},
        {"source": "database.md", "heading": "MySQL 索引"},
        {"source": "concurrency.md", "heading": "异步与事件循环"},
        {"source": "llm_rag.md", "heading": "RAG 流程"},
    ]

    score = score_results(results, GOLD)

    assert score["rank"] == 4
    assert score["hit@3"] is False
    assert score["hit@5"] is True
    assert score["mrr"] == 0.25


def test_score_results_no_hit():
    results = [
        {"source": "network.md", "heading": "HTTP 与 HTTPS"},
        {"source": "database.md", "heading": "MySQL 索引"},
    ]

    score = score_results(results, GOLD)

    assert score["rank"] is None
    assert score["hit@3"] is False
    assert score["hit@5"] is False
    assert score["mrr"] == 0.0


def test_scan_parameters_runs_expected_grid(monkeypatch):
    retrieve_calls = []
    rerank_calls = []

    def _fake_retrieve(query, **kwargs):
        retrieve_calls.append((query, kwargs["n_candidates"], kwargs["top_k"]))
        return [
            {"id": "hit", "source": "llm_rag.md", "heading": "RAG 流程", "excerpt": "RAG 流程"},
            {"id": "miss", "source": "network.md", "heading": "HTTP 与 HTTPS", "excerpt": "HTTP"},
        ]

    def _fake_rerank(query, candidates, text_key, top_k):
        rerank_calls.append((query, len(candidates), top_k))
        return candidates[:top_k]

    monkeypatch.setattr(retrieval_eval, "retrieve", _fake_retrieve)
    monkeypatch.setattr("src.rag.rerank.rerank", _fake_rerank)

    runs = scan_parameters([{"id": "case", "query": "RAG 怎么检索？", "gold": GOLD}])

    assert len(runs) == 18
    assert retrieve_calls == [
        ("RAG 怎么检索？", 30, 30),
        (f"{BGE_QUERY_PREFIX}RAG 怎么检索？", 30, 30),
    ]
    assert rerank_calls == [
        ("RAG 怎么检索？", 2, 30),
        (f"{BGE_QUERY_PREFIX}RAG 怎么检索？", 2, 30),
    ]
    assert runs[0]["config"] == {"top_k": 3, "n_candidates": 10, "query_prefix": False}


def test_evaluate_applies_query_prefix(monkeypatch):
    seen = {}

    def _fake_retrieve(query, **kwargs):
        seen["query"] = query
        return [{"source": "llm_rag.md", "heading": "RAG 流程"}]

    monkeypatch.setattr(retrieval_eval, "retrieve", _fake_retrieve)

    retrieval_eval.evaluate(
        [{"id": "one", "query": "RAG 怎么检索？", "gold": GOLD}],
        query_prefix=BGE_QUERY_PREFIX,
    )

    assert seen["query"] == f"{BGE_QUERY_PREFIX}RAG 怎么检索？"
