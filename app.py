"""Streamlit UI for jobagent."""
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import streamlit as st

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
    if not company.strip():
        st.error("请填写目标公司")
    elif not role.strip():
        st.error("请填写目标岗位")
    elif not jd.strip():
        st.error("请填写 JD 文本")
    else:
        from src.agent.graph import run_workflow
        from src.schemas.plan import JobInput

        job = JobInput(company=company.strip(), role=role.strip(), jd_text=jd.strip())
        with st.spinner("生成中，预计需要 1-3 分钟..."):
            final = run_workflow(job, out_dir.strip() or "runs/streamlit")

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
            plan_md_path = Path(out_dir) / "plan.md"
            plan_json_path = Path(out_dir) / "plan.json"

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
