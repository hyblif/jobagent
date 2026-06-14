"""Tests for RAG retriever — uses tmp_path Chroma, mocks embedding and reranker."""
import pytest

from src.rag.ingest import chunk_markdown
from src.rag.store import get_collection


QA_MD = """\
# 测试知识库

## 基础

Q1: 什么是向量检索？
A: 向量检索通过计算 embedding 向量的相似度来找到语义相近的文档，使用 HNSW 等 ANN 算法加速。

Q2: 什么是 RAG？
A: RAG（检索增强生成）将外部知识库的检索结果注入到 LLM 的 prompt 中，减少幻觉并提升答案质量。

Q3: Reranker 有什么作用？
A: Reranker 使用 CrossEncoder 对召回候选重新打分，提升 top-k 的 precision，弥补 BiEncoder 交互建模弱的缺点。
"""


@pytest.fixture
def mock_embed(monkeypatch):
    """Replace real embedding with a deterministic char-count vector."""
    import numpy as np
    from src.rag import embed as embed_module

    def _fake_load():
        class FakeModel:
            def encode(self, texts, **kwargs):
                # Simple deterministic: each text's vector is based on char counts
                vecs = []
                for t in texts:
                    # 384-dim zero vector with first dim = len(t) / 1000
                    v = np.zeros(384)
                    v[0] = len(t) / 1000.0
                    vecs.append(v)
                return np.array(vecs)
        return FakeModel()

    monkeypatch.setattr(embed_module, "_load_model", _fake_load)
    embed_module._load_model.cache_clear = lambda: None
    embed_module.get_embedding_function.cache_clear()


@pytest.fixture
def populated_collection(mock_embed, tmp_path):
    """Build a small Chroma collection with 3 Q&A chunks."""
    col = get_collection(reset=True)
    chunks = chunk_markdown(QA_MD, "test.md")
    ids = [f"test-{i}" for i in range(len(chunks))]
    docs = [c.text for c in chunks]
    metas = [c.metadata for c in chunks]
    col.add(ids=ids, documents=docs, metadatas=metas)
    return col


def test_empty_collection_returns_empty(mock_embed, tmp_path):
    from src.rag.retriever import retrieve
    # Collection exists but is empty
    get_collection(reset=True)
    results = retrieve("向量检索", n_candidates=5, top_k=3, use_rerank=False)
    assert results == []


def test_empty_collection_can_raise_for_workflow_warning(mock_embed, tmp_path):
    from src.rag.retriever import KnowledgeBaseEmptyError, retrieve
    get_collection(reset=True)
    with pytest.raises(KnowledgeBaseEmptyError, match="知识库为空"):
        retrieve("向量检索", n_candidates=5, top_k=3, use_rerank=False, raise_on_error=True)


def test_retrieve_returns_candidates(populated_collection, monkeypatch):
    """Without rerank, retrieve should return top-k candidates."""
    # Patch reranker so it's never called
    monkeypatch.setattr("src.rag.retriever.rerank", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("rerank called")))
    from src.rag.retriever import retrieve
    results = retrieve("什么是 RAG", n_candidates=10, top_k=2, use_rerank=False)
    assert 1 <= len(results) <= 2
    for r in results:
        assert "excerpt" in r
        assert r["source_type"] == "kb"
        assert r["source"] == "test.md"
        assert r["heading"] == "基础"


def test_retrieve_with_rerank(populated_collection, monkeypatch):
    """With rerank=True, rerank function should be called."""
    call_log = []

    def _fake_rerank(query, candidates, text_key, top_k):
        call_log.append(query)
        return candidates[:top_k]

    monkeypatch.setattr("src.rag.retriever.rerank", _fake_rerank)
    from src.rag.retriever import retrieve
    results = retrieve("Reranker 作用", top_k=2, use_rerank=True)
    assert len(call_log) == 1
    assert len(results) <= 2


def test_retrieve_graceful_on_exception(monkeypatch):
    """If collection query throws, return [] gracefully."""
    from src.rag import retriever as ret_module
    monkeypatch.setattr(ret_module, "get_collection", lambda: (_ for _ in ()).throw(RuntimeError("db error")))
    from src.rag.retriever import retrieve
    results = retrieve("query")
    assert results == []


def test_retrieve_can_raise_on_exception(monkeypatch):
    """Workflow mode can ask retrieve to expose collection errors."""
    from src.rag import retriever as ret_module
    monkeypatch.setattr(ret_module, "get_collection", lambda: (_ for _ in ()).throw(RuntimeError("db error")))
    from src.rag.retriever import retrieve
    with pytest.raises(RuntimeError, match="知识库连接失败"):
        retrieve("query", raise_on_error=True)
