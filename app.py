"""Streamlit UI for jobagent."""
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import streamlit as st

from src.agent.state import AgentState
from src.schemas.plan import JobInput

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


st.set_page_config(page_title="jobagent · 面试备战计划", layout="wide")
st.title("jobagent · AI 面试备战计划生成器")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("输入信息")
    company = st.text_input("目标公司", placeholder="例：字节跳动")
    role = st.text_input("目标岗位", placeholder="例：AI Agent 工程师")
    jd = st.text_area("JD 文本", height=300, placeholder="将职位描述粘贴到此处...")
    out_dir = st.text_input("输出目录", value="runs/streamlit", help="生成的 plan.json/plan.md 保存路径")
    run_btn = st.button("生成备战计划", type="primary", use_container_width=True)

with col2:
    st.subheader("生成结果")
    result_placeholder = st.empty()

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

        warnings = final.get("warnings", [])
        web_count = len(final.get("web_evidences", []))
        kb_count = len(final.get("kb_evidences", []))

        if warnings:
            st.warning("运行告警：\n" + "\n".join(f"- {w}" for w in warnings))

        with st.sidebar:
            st.subheader("证据计数")
            st.metric("Web 证据", web_count)
            st.metric("KB 证据", kb_count)

        if not final.get("plan"):
            errors = final.get("validation_errors", [])
            with col2:
                st.error("生成失败 — 计划校验未通过，输出可能不完整")
                for e in errors:
                    st.write(f"- {e}")
        else:
            plan_md_path = Path(output_dir) / "plan.md"
            plan_json_path = Path(output_dir) / "plan.json"

            with col2:
                st.success("生成完成！")

                if plan_md_path.exists():
                    md_content = plan_md_path.read_text(encoding="utf-8")
                    st.markdown(md_content)

                    dl_col1, dl_col2 = st.columns(2)
                    with dl_col1:
                        st.download_button(
                            "⬇ 下载 plan.md",
                            md_content,
                            file_name="plan.md",
                            mime="text/markdown",
                        )
                    with dl_col2:
                        if plan_json_path.exists():
                            st.download_button(
                                "⬇ 下载 plan.json",
                                plan_json_path.read_text(encoding="utf-8"),
                                file_name="plan.json",
                                mime="application/json",
                            )

                # Show evidence summary in sidebar
                all_evidences = final.get("all_evidences", [])
                if all_evidences:
                    with st.expander(f"📚 证据来源（共 {len(all_evidences)} 条）"):
                        for e in all_evidences:
                            st.write(f"**[{e.id}]** ({e.source_type}) {e.title}")
                            if e.url_or_path:
                                st.caption(e.url_or_path)
