from src.eval.retrieval_eval import score_results


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
