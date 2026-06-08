"""LangGraph node functions and routing logic for the jobagent workflow."""
import json
import os
import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from src.agent.state import AgentState
from src.llm import get_llm
from src.rag.retriever import retrieve
from src.schemas.plan import Evidence, PrepPlan
from src.tools.web_search import web_search

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "plan_prompt.md"

MAX_RETRIES = 3
MAX_WEB_EVIDENCES = 12
MAX_KB_CANDIDATES = 6


def _load_system_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def parse_json(content: str) -> dict | None:
    """Strip optional ```json fences and parse JSON."""
    content = content.strip()
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Node 1: intake
# ---------------------------------------------------------------------------

def intake_node(state: AgentState) -> dict:
    job = state["job_input"]
    jd_snippet = job.jd_text[:2000]

    system = (
        "你是面试调研助手。根据公司、岗位和 JD，生成 4 条用于联网搜索的中文检索词，"
        "覆盖：(1) 公司+岗位面试流程/面经，(2) 岗位核心技术栈考点，"
        "(3) 公司近期业务/技术方向，(4) 该岗位常见八股题方向。"
        '只输出 JSON：{"queries": ["...", "...", "...", "..."]}（json）。'
    )
    human = f"公司：{job.company}\n岗位：{job.role}\nJD：{jd_snippet}"

    try:
        llm = get_llm(temperature=0.1)
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
        data = parse_json(resp.content)
        queries = data.get("queries", []) if data else []
        queries = [q for q in queries if isinstance(q, str)][:4]
    except Exception:
        queries = []

    if not queries:
        queries = [
            f"{job.company} {job.role} 面试流程",
            f"{job.company} {job.role} 面经",
            f"{job.role} 高频考点",
            f"{job.role} 八股",
        ]

    return {"search_queries": queries}


# ---------------------------------------------------------------------------
# Node 2: research
# ---------------------------------------------------------------------------

def research_node(state: AgentState) -> dict:
    queries = state.get("search_queries", [])
    seen_urls: set[str] = set()
    results: list[dict] = []

    try:
        for q in queries:
            for item in web_search(q, max_results=5):
                url = item.get("url_or_path", "")
                if url and url in seen_urls:
                    continue
                seen_urls.add(url)
                results.append(item)
                if len(results) >= MAX_WEB_EVIDENCES:
                    break
            if len(results) >= MAX_WEB_EVIDENCES:
                break
    except Exception:
        pass

    return {"web_evidences": results}


# ---------------------------------------------------------------------------
# Node 3: retrieve
# ---------------------------------------------------------------------------

def retrieve_node(state: AgentState) -> dict:
    queries = state.get("search_queries", [])
    seen_ids: set[str] = set()
    candidates: list[dict] = []

    for q in queries:
        for item in retrieve(q, n_candidates=20, top_k=5, use_rerank=False):
            cid = item.get("id", "")
            if cid in seen_ids:
                continue
            seen_ids.add(cid)
            candidates.append(item)

    return {"kb_evidences": candidates}


# ---------------------------------------------------------------------------
# Node 4: rerank — THE ONLY place evidence IDs are assigned
# ---------------------------------------------------------------------------

def rerank_node(state: AgentState) -> dict:
    job = state["job_input"]
    raw_web: list[dict] = state.get("web_evidences", [])
    raw_kb: list[dict] = state.get("kb_evidences", [])

    # Rerank KB candidates against a composite query
    if raw_kb:
        from src.rag.rerank import rerank as _rerank
        kb_query = f"{job.role} {job.company} 面试 高频考点"
        reranked_kb = _rerank(kb_query, raw_kb, text_key="excerpt", top_k=MAX_KB_CANDIDATES)
    else:
        reranked_kb = []

    # Take top-N web evidences (already sorted by Tavily score)
    top_web = sorted(
        [w for w in raw_web if w.get("url_or_path")],
        key=lambda x: x.get("score") or 0,
        reverse=True,
    )[:MAX_KB_CANDIDATES]

    # Assign stable IDs
    web_evidences: list[Evidence] = []
    for i, w in enumerate(top_web, 1):
        web_evidences.append(
            Evidence(
                id=f"web-{i}",
                source_type="web",
                title=w.get("title") or "(untitled)",
                url_or_path=w.get("url_or_path", ""),
                excerpt=w.get("excerpt", ""),
                score=w.get("score"),
            )
        )

    kb_evidences: list[Evidence] = []
    for i, k in enumerate(reranked_kb, 1):
        kb_evidences.append(
            Evidence(
                id=f"kb-{i}",
                source_type="kb",
                title=k.get("title", "knowledge base"),
                url_or_path=k.get("url_or_path", ""),
                excerpt=k.get("excerpt", ""),
                score=k.get("score"),
            )
        )

    all_evidences = web_evidences + kb_evidences
    return {
        "web_evidences": web_evidences,
        "kb_evidences": kb_evidences,
        "all_evidences": all_evidences,
    }


# ---------------------------------------------------------------------------
# Node 5: plan
# ---------------------------------------------------------------------------

def _build_evidence_block(evidences: list[Evidence]) -> str:
    lines = []
    for e in evidences:
        lines.append(f"[{e.id}] ({e.source_type}) {e.title}")
        lines.append(e.excerpt[:800])
        lines.append(f"来源: {e.url_or_path}")
        lines.append("")
    return "\n".join(lines)


def plan_node(state: AgentState) -> dict:
    job = state["job_input"]
    all_evidences: list[Evidence] = state.get("all_evidences", [])
    validation_errors: list[str] = state.get("validation_errors", [])

    system_prompt = _load_system_prompt()
    evidence_block = _build_evidence_block(all_evidences)

    user_parts = [
        f"【公司】{job.company}",
        f"【岗位】{job.role}",
        f"【JD】\n{job.jd_text[:2000]}",
        "",
        "【证据】",
        evidence_block if evidence_block.strip() else "（无检索结果，请基于通用面试准备建议生成计划）",
        "",
        "请严格依照上述证据和输出格式，生成面试备战计划 JSON（json）。",
        "所有 source_ids 只能引用上面【证据】中出现的 id。",
    ]

    if validation_errors:
        user_parts.append("")
        user_parts.append("【修复要求】上一次生成存在以下问题，请修正后重新输出完整 JSON（json）：")
        for err in validation_errors:
            user_parts.append(f"- {err}")

    user_message = "\n".join(user_parts)

    try:
        llm = get_llm(temperature=0.2)
        resp = llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]
        )
        plan_json = parse_json(resp.content)
        if plan_json is None:
            return {
                "plan_json": None,
                "validation_errors": ["LLM 输出无法解析为 JSON，请重新生成"],
            }
        return {"plan_json": plan_json, "validation_errors": []}
    except Exception as exc:
        return {
            "plan_json": None,
            "validation_errors": [f"LLM 调用失败：{exc}"],
        }


# ---------------------------------------------------------------------------
# Node 6: validate
# ---------------------------------------------------------------------------

def validate_node(state: AgentState) -> dict:
    plan_json = state.get("plan_json")
    retry_count = state.get("retry_count", 0)
    all_evidences: list[Evidence] = state.get("all_evidences", [])
    valid_ids = {e.id for e in all_evidences}

    errors: list[str] = []

    if not plan_json:
        errors.append("plan_json 为空，无法校验")
        return {"validation_errors": errors, "retry_count": retry_count + 1, "plan": None}

    # Stage A: Pydantic schema validation
    try:
        plan = PrepPlan(**plan_json)
    except ValidationError as ve:
        for err in ve.errors():
            loc = ".".join(str(l) for l in err["loc"])
            errors.append(f"Schema 错误 [{loc}]: {err['msg']}")
        return {"validation_errors": errors, "retry_count": retry_count + 1, "plan": None}

    # Stage B: source_id existence check
    all_section_ids: set[str] = set()
    for section in [
        plan.interview_overview,
        plan.focus_areas,
        plan.high_frequency_topics,
        plan.q_and_a,
        plan.action_plan,
    ]:
        for item in section:
            for sid in item.get("source_ids", []):
                all_section_ids.add(sid)
                if valid_ids and sid not in valid_ids:
                    errors.append(f"未知 source_id: {sid}（不在证据列表中）")

    # Stage C: completeness check
    if len(plan.q_and_a) == 0:
        errors.append("q_and_a 为空，至少需要 1 道题目")
    if len(plan.focus_areas) == 0:
        errors.append("focus_areas 为空，至少需要 1 个备考方向")

    if errors:
        return {"validation_errors": errors, "retry_count": retry_count + 1, "plan": None}

    # Backfill citations: ensure every cited id has an entry in plan.citations
    existing_citation_ids = {c.get("id") for c in plan.citations}
    evidence_map = {e.id: e for e in all_evidences}
    citations = list(plan.citations)
    for sid in all_section_ids:
        if sid not in existing_citation_ids and sid in evidence_map:
            e = evidence_map[sid]
            citations.append(
                {
                    "id": e.id,
                    "title": e.title,
                    "url_or_path": e.url_or_path,
                    "excerpt": e.excerpt[:300],
                }
            )
    plan_json["citations"] = citations

    # Re-parse with backfilled citations
    try:
        plan = PrepPlan(**plan_json)
    except ValidationError:
        pass  # citations backfill shouldn't fail; proceed with original

    return {"plan": plan, "plan_json": plan_json, "validation_errors": []}


# ---------------------------------------------------------------------------
# Node 7: render_node
# ---------------------------------------------------------------------------

def _render_markdown(plan: PrepPlan, job, citations_map: dict) -> str:
    lines = [f"# {job.company} · {job.role} 面试备战计划\n"]

    def ref(source_ids):
        if not source_ids:
            return ""
        return " " + "".join(f"[{sid}]" for sid in source_ids)

    lines.append("## 一、面试流程概览\n")
    for item in plan.interview_overview:
        phase = item.get("phase", "")
        desc = item.get("description", "")
        sids = item.get("source_ids", [])
        lines.append(f"- **{phase}**：{desc}{ref(sids)}")
    lines.append("")

    lines.append("## 二、核心备考方向\n")
    for item in plan.focus_areas:
        area = item.get("area", "")
        imp = item.get("importance", "")
        sids = item.get("source_ids", [])
        lines.append(f"- **{area}**（重要度：{imp}）{ref(sids)}")
    lines.append("")

    lines.append("## 三、高频考点\n")
    for item in plan.high_frequency_topics:
        topic = item.get("topic", "")
        sids = item.get("source_ids", [])
        lines.append(f"- {topic}{ref(sids)}")
    lines.append("")

    lines.append("## 四、Q&A 备战题库\n")
    for i, item in enumerate(plan.q_and_a, 1):
        q = item.get("question", "")
        a = item.get("answer_outline", "")
        sids = item.get("source_ids", [])
        lines.append(f"### Q{i}: {q}{ref(sids)}")
        lines.append(f"**答题要点**：{a}\n")
    lines.append("")

    lines.append("## 五、行动计划\n")
    for item in plan.action_plan:
        week = item.get("week", "")
        tasks = item.get("tasks", [])
        sids = item.get("source_ids", [])
        lines.append(f"### {week}{ref(sids)}")
        for t in tasks:
            lines.append(f"- {t}")
        lines.append("")

    lines.append("## 引用来源\n")
    for c in plan.citations:
        cid = c.get("id", "")
        title = c.get("title", "")
        url = c.get("url_or_path", "")
        lines.append(f"- [{cid}] **{title}** — {url}")
    lines.append("")

    return "\n".join(lines)


def render_node(state: AgentState) -> dict:
    import json as _json
    from pathlib import Path

    output_dir = state.get("output_dir", "runs/output")
    plan: PrepPlan | None = state.get("plan")
    plan_json = state.get("plan_json")
    all_evidences = state.get("all_evidences", [])
    job = state["job_input"]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save evidence
    evidence_data = [e.model_dump() for e in all_evidences]
    (Path(output_dir) / "evidence.json").write_text(
        _json.dumps(evidence_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Save plan JSON
    if plan_json:
        (Path(output_dir) / "plan.json").write_text(
            _json.dumps(plan_json, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # Render and save Markdown
    if plan:
        citations_map = {c.get("id"): c for c in plan.citations}
        md = _render_markdown(plan, job, citations_map)
    else:
        validation_errors = state.get("validation_errors", [])
        md = (
            f"# {job.company} · {job.role} 面试备战计划\n\n"
            "> **警告**：计划生成未完全通过校验，输出可能不完整。\n\n"
            "**校验错误**：\n"
            + "\n".join(f"- {e}" for e in validation_errors)
            + "\n\n请检查 `plan.json` 获取原始输出。\n"
        )

    (Path(output_dir) / "plan.md").write_text(md, encoding="utf-8")

    return {}


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def router_after_validate(state: AgentState) -> str:
    errors = state.get("validation_errors", [])
    retry_count = state.get("retry_count", 0)
    if errors and retry_count < MAX_RETRIES:
        return "plan"
    return "render"
