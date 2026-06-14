"""Tests for node logic — LLM and external calls fully mocked."""
import json

import pytest

from src.agent.nodes import (
    MAX_RETRIES,
    plan_node,
    router_after_validate,
    research_node,
    retrieve_node,
    validate_node,
    rerank_node,
    intake_node,
)
from src.schemas.plan import Evidence, JobInput


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_job():
    return JobInput(company="测试公司", role="测试岗位", jd_text="需要 Python、RAG、LangGraph 和 Redis 经验")


def _base_state(**overrides):
    state = {
        "job_input": _make_job(),
        "search_queries": [],
        "kb_query": "",
        "web_evidences": [],
        "kb_evidences": [],
        "all_evidences": [],
        "plan_json": None,
        "plan": None,
        "validation_errors": [],
        "warnings": [],
        "retry_count": 0,
        "output_dir": "/tmp/test",
    }
    state.update(overrides)
    return state


def _make_evidences(n_web=2, n_kb=2) -> list[Evidence]:
    evs = []
    for i in range(1, n_web + 1):
        evs.append(Evidence(id=f"web-{i}", source_type="web", title=f"Web {i}",
                            url_or_path=f"https://example.com/{i}", excerpt="web content"))
    for i in range(1, n_kb + 1):
        evs.append(Evidence(id=f"kb-{i}", source_type="kb", title=f"KB {i}",
                            url_or_path=f"kb_file_{i}.md", excerpt="kb content"))
    return evs


def _valid_plan_json(evidences):
    """Build a minimal valid plan_json referencing only real evidence ids."""
    sid = evidences[0].id if evidences else "web-1"
    return {
        "interview_overview": [{"phase": "一面", "description": "技术面", "source_ids": [sid]}],
        "focus_areas": [{"area": "Python", "importance": "高", "source_ids": [sid]}],
        "high_frequency_topics": [{"topic": "GIL", "source_ids": [sid]}],
        "q_and_a": [{"question": "什么是 GIL？", "answer_outline": "全局锁，单线程执行字节码", "source_ids": [sid]}],
        "action_plan": [{"week": "第1周", "tasks": ["复习 Python"], "source_ids": [sid]}],
        "citations": [{"id": sid, "title": evidences[0].title, "url_or_path": evidences[0].url_or_path, "excerpt": "..."}],
    }


# ---------------------------------------------------------------------------
# intake_node fallback
# ---------------------------------------------------------------------------

def test_intake_node_fallback_on_llm_failure(monkeypatch):
    """When LLM fails, intake_node should return 4 template queries."""
    def _fail(*args, **kwargs):
        raise RuntimeError("LLM unavailable")

    monkeypatch.setattr("src.agent.nodes.get_llm", _fail)

    state = _base_state()
    result = intake_node(state)
    assert len(result["search_queries"]) == 4
    assert all("测试" in q or "岗位" in q for q in result["search_queries"])
    assert result["kb_query"]
    assert "Python" in result["kb_query"] or "python" in result["kb_query"].lower()
    assert any("intake 降级为模板检索词" in w for w in result["warnings"])


def test_research_node_warns_on_tavily_failure(monkeypatch):
    """Tavily failure should be visible in workflow state."""
    def _fail(*args, **kwargs):
        raise RuntimeError("bad tavily key")

    monkeypatch.setattr("src.agent.nodes.web_search", _fail)
    result = research_node(_base_state(search_queries=["测试 查询"]))
    assert result["web_evidences"] == []
    assert any("Tavily 搜索失败" in w and "web 证据 0 条" in w for w in result["warnings"])


def test_retrieve_node_warns_on_empty_kb(monkeypatch):
    """Knowledge-base failures should be captured as warnings."""
    def _fail(*args, **kwargs):
        raise RuntimeError("知识库为空，建议先 build_index")

    monkeypatch.setattr("src.agent.nodes.retrieve", _fail)
    result = retrieve_node(_base_state(search_queries=["RAG 面试"]))
    assert result["kb_evidences"] == []
    assert any("知识库检索失败" in w and "build_index" in w for w in result["warnings"])


def test_plan_node_warns_on_llm_failure(monkeypatch):
    """LLM call failures should remain validation errors and warnings."""
    class FailingLLM:
        def invoke(self, messages):
            raise RuntimeError("llm down")

    monkeypatch.setattr("src.agent.nodes.get_llm", lambda temperature=0.2: FailingLLM())
    result = plan_node(_base_state(all_evidences=[]))
    assert result["plan_json"] is None
    assert any("LLM 调用失败" in e for e in result["validation_errors"])
    assert any("LLM 调用失败" in w for w in result["warnings"])


# ---------------------------------------------------------------------------
# rerank_node ID assignment
# ---------------------------------------------------------------------------

def test_rerank_node_assigns_stable_ids(monkeypatch):
    """rerank_node must assign web-N and kb-N ids."""
    def _fake_rerank(query, candidates, text_key, top_k):
        return candidates[:top_k]

    monkeypatch.setattr("src.rag.rerank.rerank", _fake_rerank, raising=False)

    web_raw = [
        {"source_type": "web", "title": "W1", "url_or_path": "https://a.com", "excerpt": "e1", "score": 0.9},
        {"source_type": "web", "title": "W2", "url_or_path": "https://b.com", "excerpt": "e2", "score": 0.8},
    ]
    kb_raw = [
        {"id": "chroma-001", "source_type": "kb", "title": "KB1", "url_or_path": "os.md", "excerpt": "e3", "score": 0.7},
    ]
    state = _base_state(
        search_queries=[],
        kb_query="测试岗位 Python RAG 面试 高频考点",
        web_evidences=web_raw,
        kb_evidences=kb_raw,
        output_dir="/tmp",
    )

    result = rerank_node(state)
    web_evs = result["web_evidences"]
    kb_evs = result["kb_evidences"]

    # Check IDs follow the web-N / kb-N pattern
    for i, e in enumerate(web_evs, 1):
        assert e.id == f"web-{i}"
    for i, e in enumerate(kb_evs, 1):
        assert e.id == f"kb-{i}"

    all_ids = {e.id for e in result["all_evidences"]}
    assert all_ids == {e.id for e in web_evs} | {e.id for e in kb_evs}


def test_rerank_node_uses_kb_query(monkeypatch):
    """rerank_node should pass the intake-generated kb_query to the CrossEncoder reranker."""
    captured = {}

    def _fake_rerank(query, candidates, text_key, top_k):
        captured["query"] = query
        return candidates[:top_k]

    monkeypatch.setattr("src.rag.rerank.rerank", _fake_rerank, raising=False)
    kb_raw = [
        {"id": "chroma-001", "source_type": "kb", "title": "KB1", "url_or_path": "rag.md", "excerpt": "LangGraph RAG", "score": 0.7},
    ]
    state = _base_state(
        kb_query="AI Agent LangGraph RAG Redis 面试 高频考点",
        kb_evidences=kb_raw,
    )
    rerank_node(state)
    assert captured["query"] == "AI Agent LangGraph RAG Redis 面试 高频考点"


# ---------------------------------------------------------------------------
# validate_node
# ---------------------------------------------------------------------------

def test_validate_node_passes_valid_plan():
    evidences = _make_evidences(2, 2)
    state = {
        "job_input": _make_job(),
        "search_queries": [], "kb_query": "", "web_evidences": [], "kb_evidences": [],
        "all_evidences": evidences,
        "plan_json": _valid_plan_json(evidences),
        "plan": None, "validation_errors": [], "warnings": [], "retry_count": 0, "output_dir": "/tmp",
    }
    result = validate_node(state)
    assert result["validation_errors"] == []
    assert result["plan"] is not None


def test_validate_node_detects_unknown_source_id():
    evidences = _make_evidences(1, 1)
    plan_json = _valid_plan_json(evidences)
    # Inject a fake source_id
    plan_json["focus_areas"][0]["source_ids"] = ["nonexistent-99"]

    state = {
        "job_input": _make_job(),
        "search_queries": [], "kb_query": "", "web_evidences": [], "kb_evidences": [],
        "all_evidences": evidences,
        "plan_json": plan_json,
        "plan": None, "validation_errors": [], "warnings": [], "retry_count": 0, "output_dir": "/tmp",
    }
    result = validate_node(state)
    assert any("nonexistent-99" in e for e in result["validation_errors"])
    assert result["retry_count"] == 1


def test_validate_node_increments_retry_count():
    evidences = _make_evidences(1, 1)
    plan_json = _valid_plan_json(evidences)
    plan_json["q_and_a"] = []  # trigger completeness error

    state = {
        "job_input": _make_job(),
        "search_queries": [], "kb_query": "", "web_evidences": [], "kb_evidences": [],
        "all_evidences": evidences,
        "plan_json": plan_json,
        "plan": None, "validation_errors": [], "warnings": [], "retry_count": 1, "output_dir": "/tmp",
    }
    result = validate_node(state)
    assert result["retry_count"] == 2


def test_validate_node_schema_error():
    evidences = _make_evidences(1, 0)
    # Missing required fields
    state = {
        "job_input": _make_job(),
        "search_queries": [], "kb_query": "", "web_evidences": [], "kb_evidences": [],
        "all_evidences": evidences,
        "plan_json": {"interview_overview": []},  # missing 4 required fields
        "plan": None, "validation_errors": [], "warnings": [], "retry_count": 0, "output_dir": "/tmp",
    }
    result = validate_node(state)
    assert len(result["validation_errors"]) > 0


# ---------------------------------------------------------------------------
# router_after_validate
# ---------------------------------------------------------------------------

def test_router_routes_to_plan_when_errors_and_retries_left():
    state = {"validation_errors": ["some error"], "retry_count": 1}
    assert router_after_validate(state) == "plan"


def test_router_routes_to_render_when_no_errors():
    state = {"validation_errors": [], "retry_count": 0}
    assert router_after_validate(state) == "render"


def test_router_routes_to_render_when_max_retries_reached():
    state = {"validation_errors": ["error"], "retry_count": MAX_RETRIES}
    assert router_after_validate(state) == "render"
