"""Integration test: full workflow with all external calls mocked.
Tests the validate->plan retry loop: LLM first returns unparseable content, then valid JSON.
"""
import json
from pathlib import Path

import pytest

from src.schemas.plan import JobInput


def _valid_plan_json():
    """Valid plan JSON with empty source_ids (no evidence required to validate)."""
    return {
        "interview_overview": [{"phase": "一面", "description": "技术面", "source_ids": []}],
        "focus_areas": [{"area": "Python", "importance": "高", "source_ids": []}],
        "high_frequency_topics": [{"topic": "GIL", "source_ids": []}],
        "q_and_a": [{"question": "GIL 是什么？", "answer_outline": "全局解释器锁；单线程字节码执行", "source_ids": []}],
        "action_plan": [{"week": "第1周", "tasks": ["复习 Python 并发"], "source_ids": []}],
        "citations": [],
    }


@pytest.fixture
def patched_workflow(monkeypatch, tmp_path):
    """Patch all heavy external calls.
    LLM strategy: intake returns 4 queries; plan LLM returns non-JSON on 1st call (triggers
    validation error → retry), then valid JSON on 2nd call.
    """
    call_count = {"n": 0}

    class FakeLLM:
        def invoke(self, messages):
            class Resp:
                content = '{"queries": ["q1", "q2", "q3", "q4"]}'
            return Resp()

    class FakePlanLLM:
        def invoke(self, messages):
            call_count["n"] += 1
            class Resp:
                pass
            r = Resp()
            if call_count["n"] == 1:
                # First call: NOT valid JSON → parse_json returns None → validation error → retry
                r.content = "这不是有效的 JSON，请忽略。"
            else:
                r.content = json.dumps(_valid_plan_json(), ensure_ascii=False)
            return r

    import src.agent.nodes as nodes_module

    def _fake_get_llm(temperature=0.2):
        if temperature == 0.1:
            return FakeLLM()
        return FakePlanLLM()

    monkeypatch.setattr(nodes_module, "get_llm", _fake_get_llm)

    # Patch web_search and retrieve to return empty (no network, no models)
    monkeypatch.setattr(nodes_module, "web_search", lambda *a, **kw: [])
    monkeypatch.setattr(nodes_module, "retrieve", lambda *a, **kw: [])

    # Patch rerank_node in BOTH graph and nodes namespace so the compiled graph uses it
    import src.agent.graph as graph_module

    def _noop_rerank_node(state):
        return {"web_evidences": [], "kb_evidences": [], "all_evidences": []}

    monkeypatch.setattr(nodes_module, "rerank_node", _noop_rerank_node)
    monkeypatch.setattr(graph_module, "rerank_node", _noop_rerank_node)

    return {"tmp_path": tmp_path, "call_count": call_count}


def test_workflow_produces_output_files(patched_workflow, tmp_path):
    from src.agent.graph import run_workflow

    out_dir = str(tmp_path / "test_run")
    job = JobInput(company="测试公司", role="测试岗位", jd_text="测试 JD 内容")
    run_workflow(job, out_dir)

    assert Path(out_dir, "plan.json").exists(), "plan.json should be created"
    assert Path(out_dir, "plan.md").exists(), "plan.md should be created"
    assert Path(out_dir, "evidence.json").exists(), "evidence.json should be created"


def test_workflow_retry_loop_executes(patched_workflow):
    """plan LLM called twice: first non-JSON triggers validate error + retry, second succeeds."""
    from src.agent.graph import run_workflow

    job = JobInput(company="测试公司", role="测试岗位", jd_text="测试 JD")
    run_workflow(job, "/tmp/retry_test")

    assert patched_workflow["call_count"]["n"] == 2, (
        f"Expected 2 LLM plan calls, got {patched_workflow['call_count']['n']}"
    )


def test_workflow_final_state_has_plan(patched_workflow, tmp_path):
    from src.agent.graph import run_workflow

    job = JobInput(company="测试公司", role="测试岗位", jd_text="测试 JD")
    final = run_workflow(job, str(tmp_path / "plan_check"))

    assert final.get("plan") is not None
    assert final["validation_errors"] == []


def test_plan_md_contains_key_sections(patched_workflow, tmp_path):
    from src.agent.graph import run_workflow

    out_dir = str(tmp_path / "sections_check")
    job = JobInput(company="测试公司", role="测试岗位", jd_text="测试 JD")
    run_workflow(job, out_dir)

    md = Path(out_dir, "plan.md").read_text(encoding="utf-8")
    assert "面试流程" in md
    assert "核心备考方向" in md
    assert "Q&A" in md
    assert "行动计划" in md
    assert "引用来源" in md
