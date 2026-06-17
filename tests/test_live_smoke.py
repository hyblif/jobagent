"""Live end-to-end smoke test.

This test is intentionally skipped by default. Enable it with RUN_LIVE_TESTS=1
and real API keys in .env to verify the actual Tavily + DeepSeek workflow.
"""
import json
import os
import time
from pathlib import Path

import pytest
from dotenv import load_dotenv

from src.schemas.plan import JobInput, PrepPlan

_REPO_ROOT = Path(__file__).parent.parent


@pytest.mark.live
def test_live_workflow_smoke(monkeypatch, tmp_path):
    if os.environ.get("RUN_LIVE_TESTS") != "1":
        pytest.skip("Set RUN_LIVE_TESTS=1 to run the live smoke test.")

    # no_network autouse fixture deletes keys from the environment; restore from .env
    load_dotenv(override=True)

    missing = [k for k in ("DEEPSEEK_API_KEY", "TAVILY_API_KEY") if not os.environ.get(k)]
    if missing:
        pytest.skip(f"Missing live API configuration: {', '.join(missing)}")

    persist_dir = _REPO_ROOT / ".chroma/jobagent"
    if not persist_dir.exists():
        pytest.skip(
            "Missing .chroma/jobagent; build it with "
            "`uv run python scripts/build_index.py --data-dir data/baguwen "
            "--persist-dir .chroma/jobagent`."
        )

    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(persist_dir))
    from src.llm import get_llm
    from src.rag import store

    store.get_client.cache_clear()
    get_llm.cache_clear()

    from src.agent.graph import run_workflow

    jd_text = (_REPO_ROOT / "examples/jd_agent_engineer.txt").read_text(encoding="utf-8")
    job = JobInput(company="阿里巴巴", role="Agent 工程师", jd_text=jd_text)
    output_dir = tmp_path / "live_smoke"

    started = time.perf_counter()
    final_state = run_workflow(job, str(output_dir))
    elapsed_seconds = time.perf_counter() - started

    plan_path = output_dir / "plan.json"
    md_path = output_dir / "plan.md"
    evidence_path = output_dir / "evidence.json"

    assert plan_path.exists()
    assert md_path.exists()
    assert evidence_path.exists()

    plan = PrepPlan(**json.loads(plan_path.read_text(encoding="utf-8")))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))

    assert final_state.get("plan") is not None
    assert plan.q_and_a
    assert plan.focus_areas
    assert evidence
    assert elapsed_seconds < 180

    warnings = final_state.get("warnings", [])
    assert not any("Tavily 搜索失败" in warning for warning in warnings)
    assert not any("LLM 调用失败" in warning for warning in warnings)
