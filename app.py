"""Streamlit UI for jobagent."""
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import streamlit as st

from src.agent.state import AgentState
from src.schemas.plan import Evidence, JobInput

WORKFLOW_STEPS = [
    ("intake", "解析输入与生成检索词"),
    ("research", "联网调研岗位与公司信息"),
    ("retrieve", "检索本地知识库"),
    ("rerank", "重排并编号证据"),
    ("generate", "生成结构化备战计划"),
    ("validate", "校验计划与引用"),
    ("render", "保存 Markdown / JSON 结果"),
]

STEP_LABELS = dict(WORKFLOW_STEPS)
STEP_INDEX = {node: index for index, (node, _) in enumerate(WORKFLOW_STEPS, start=1)}
SAMPLE_JD_PATH = Path("examples/jd_agent_engineer.txt")


def _load_sample_jd() -> str:
    return SAMPLE_JD_PATH.read_text(encoding="utf-8")


def _apply_sample_input() -> None:
    st.session_state["company"] = "字节跳动"
    st.session_state["role"] = "AI Agent 工程师"
    st.session_state["jd"] = _load_sample_jd()
    st.session_state["out_dir"] = "runs/streamlit"


def _initial_state(job_input: JobInput, output_dir: str) -> AgentState:
    return {
        "job_input": job_input,
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
        "output_dir": output_dir,
    }


def _run_workflow_with_progress(job: JobInput, output_dir: str) -> dict:
    from src.agent.graph import build_graph

    state: dict = _initial_state(job, output_dir)
    graph = build_graph()

    progress = st.progress(0, text="准备启动 workflow")
    with st.status("准备启动 workflow", expanded=True) as status:
        completed_nodes: set[str] = set()
        for event in graph.stream(
            state,
            config={"recursion_limit": 25},
            stream_mode="updates",
        ):
            for node_name, update in event.items():
                if update:
                    state.update(update)

                completed_nodes.add(node_name)
                step_no = STEP_INDEX.get(node_name, len(completed_nodes))
                total_steps = len(WORKFLOW_STEPS)
                label = STEP_LABELS.get(node_name, node_name)
                retry_count = state.get("retry_count", 0)
                retry_suffix = (
                    f"（第 {retry_count + 1} 次生成尝试）"
                    if node_name == "generate" and retry_count
                    else ""
                )

                progress.progress(
                    min(step_no / total_steps, 1.0),
                    text=f"{step_no}/{total_steps} {label}",
                )
                status.write(f"完成：{label}{retry_suffix}")
                status.update(label=f"已完成：{label}", state="running")

        progress.progress(1.0, text="7/7 workflow 完成")
        status.update(label="workflow 完成", state="complete", expanded=False)

    return state


def _format_score(score: float | None) -> str:
    if score is None:
        return "未提供"
    return f"{score:.4f}"


def _source_markdown(source: str) -> str:
    if not source:
        return "来源：未提供"
    if source.startswith(("http://", "https://")):
        return f"来源：[{source}]({source})"
    return f"来源：`{source}`"


def _render_evidence_items(evidences: list[Evidence], empty_text: str) -> None:
    if not evidences:
        st.info(empty_text)
        return

    for evidence in evidences:
        with st.expander(f"[{evidence.id}] {evidence.title}", expanded=False):
            st.markdown(f"**引用 ID：`[{evidence.id}]`**")
            st.caption(
                f"类型：{evidence.source_type.upper()} | 分数：{_format_score(evidence.score)}"
            )
            st.markdown(_source_markdown(evidence.url_or_path))
            if evidence.excerpt:
                st.write(evidence.excerpt[:500])


def _render_evidence_panel(web_evidences: list[Evidence], kb_evidences: list[Evidence]) -> None:
    total = len(web_evidences) + len(kb_evidences)
    st.subheader("证据面板")
    st.caption("计划正文中的 `[web-N]` / `[kb-N]` 引用 ID 与这里一一对应。")
    st.metric("证据总数", total)
    count_col1, count_col2 = st.columns(2)
    count_col1.metric("Web", len(web_evidences))
    count_col2.metric("KB", len(kb_evidences))

    web_tab, kb_tab = st.tabs(
        [f"Web ({len(web_evidences)})", f"KB ({len(kb_evidences)})"]
    )
    with web_tab:
        _render_evidence_items(web_evidences, "暂无 Web 证据")
    with kb_tab:
        _render_evidence_items(kb_evidences, "暂无 KB 证据")


def _read_output_file(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _render_output_downloads(output_dir: str) -> None:
    output_path = Path(output_dir)
    plan_md = _read_output_file(output_path / "plan.md")
    plan_json = _read_output_file(output_path / "plan.json")
    evidence_json = _read_output_file(output_path / "evidence.json")

    st.subheader("下载结果")
    download_cols = st.columns(3)
    with download_cols[0]:
        st.download_button(
            "下载 plan.md",
            plan_md or "",
            file_name="plan.md",
            mime="text/markdown",
            disabled=plan_md is None,
            use_container_width=True,
        )
    with download_cols[1]:
        st.download_button(
            "下载 plan.json",
            plan_json or "",
            file_name="plan.json",
            mime="application/json",
            disabled=plan_json is None,
            use_container_width=True,
        )
    with download_cols[2]:
        st.download_button(
            "下载 evidence.json",
            evidence_json or "",
            file_name="evidence.json",
            mime="application/json",
            disabled=evidence_json is None,
            use_container_width=True,
        )

    missing = [
        name
        for name, content in {
            "plan.md": plan_md,
            "plan.json": plan_json,
            "evidence.json": evidence_json,
        }.items()
        if content is None
    ]
    if missing:
        st.caption("尚未生成：" + "、".join(missing))


def _render_plan_summary(plan) -> None:
    if not plan:
        return
    metric_cols = st.columns(5)
    metric_cols[0].metric("流程", len(plan.interview_overview))
    metric_cols[1].metric("方向", len(plan.focus_areas))
    metric_cols[2].metric("考点", len(plan.high_frequency_topics))
    metric_cols[3].metric("Q&A", len(plan.q_and_a))
    metric_cols[4].metric("引用", len(plan.citations))


def _render_run_alerts(final: dict) -> None:
    warnings = final.get("warnings", [])
    validation_errors = final.get("validation_errors", [])
    web_count = len(final.get("web_evidences", []))
    kb_count = len(final.get("kb_evidences", []))

    count_cols = st.columns(2)
    count_cols[0].metric("Web 证据", web_count)
    count_cols[1].metric("KB 证据", kb_count)

    if warnings:
        st.warning("运行告警：\n" + "\n".join(f"- {w}" for w in warnings))

    if validation_errors and not final.get("plan"):
        st.error("计划校验未通过，已保存可用的中间产物。")
        with st.expander("校验错误详情", expanded=True):
            for error in validation_errors:
                st.write(f"- {error}")


def _render_debug_state(final: dict) -> None:
    with st.expander("运行状态 JSON", expanded=False):
        safe_state = {
            "search_queries": final.get("search_queries", []),
            "kb_query": final.get("kb_query", ""),
            "warnings": final.get("warnings", []),
            "validation_errors": final.get("validation_errors", []),
            "retry_count": final.get("retry_count", 0),
            "web_evidence_count": len(final.get("web_evidences", [])),
            "kb_evidence_count": len(final.get("kb_evidences", [])),
        }
        st.code(json.dumps(safe_state, ensure_ascii=False, indent=2), language="json")


st.set_page_config(page_title="jobagent · 面试备战计划", layout="wide")
st.title("jobagent · AI 面试备战计划生成器")

if "out_dir" not in st.session_state:
    st.session_state["out_dir"] = "runs/streamlit"

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("输入信息")
    if st.button("填充示例 JD", use_container_width=True):
        _apply_sample_input()
    company = st.text_input("目标公司", key="company", placeholder="例：字节跳动")
    role = st.text_input("目标岗位", key="role", placeholder="例：AI Agent 工程师")
    jd = st.text_area(
        "JD 文本",
        key="jd",
        height=300,
        placeholder="将职位描述粘贴到此处...",
    )
    out_dir = st.text_input(
        "输出目录",
        key="out_dir",
        help="生成的 plan.json/plan.md/evidence.json 保存路径",
    )
    run_btn = st.button("生成备战计划", type="primary", use_container_width=True)

with col2:
    st.subheader("生成结果")

if run_btn:
    output_dir = out_dir.strip() or "runs/streamlit"
    if not company.strip():
        st.error("请填写目标公司")
    elif not role.strip():
        st.error("请填写目标岗位")
    elif not jd.strip():
        st.error("请填写 JD 文本")
    else:
        job = JobInput(company=company.strip(), role=role.strip(), jd_text=jd.strip())
        final = _run_workflow_with_progress(job, output_dir)

        web_evidences = final.get("web_evidences", [])
        kb_evidences = final.get("kb_evidences", [])

        _render_run_alerts(final)

        with st.sidebar:
            _render_evidence_panel(web_evidences, kb_evidences)

        plan_md_path = Path(output_dir) / "plan.md"

        if not final.get("plan"):
            with col2:
                st.error("生成失败：计划校验未通过。")
                _render_output_downloads(output_dir)
                _render_debug_state(final)
        else:
            with col2:
                st.success("生成完成！")
                _render_plan_summary(final["plan"])
                _render_output_downloads(output_dir)

                if plan_md_path.exists():
                    md_content = plan_md_path.read_text(encoding="utf-8")
                    st.subheader("计划预览")
                    st.markdown(md_content)
