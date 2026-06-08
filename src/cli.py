"""CLI entry point: python -m src.cli plan --company ... --role ... --jd-file ... --out ..."""
import sys
from pathlib import Path


def _load_env():
    from dotenv import load_dotenv
    load_dotenv()


import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


@click.group()
def cli():
    """jobagent — AI 面试备战计划生成器"""


@cli.command()
@click.option("--company", required=True, help="目标公司名称")
@click.option("--role", required=True, help="目标岗位名称")
@click.option("--jd-file", type=click.Path(exists=True), default=None, help="JD 文本文件路径")
@click.option("--jd-text", default=None, help="直接传入 JD 文本（与 --jd-file 二选一）")
@click.option("--out", default="runs/demo", show_default=True, help="输出目录")
def plan(company: str, role: str, jd_file: str | None, jd_text: str | None, out: str):
    """生成面试备战计划并保存到 --out 目录。"""
    _load_env()

    from src.agent.graph import build_graph
    from src.schemas.plan import JobInput

    if not jd_text and not jd_file:
        console.print("[red]错误：请通过 --jd-file 或 --jd-text 提供 JD 内容[/red]")
        sys.exit(1)

    if jd_file:
        jd_text = Path(jd_file).read_text(encoding="utf-8")

    job = JobInput(company=company, role=role, jd_text=jd_text)

    console.print(
        Panel(
            f"[bold]公司[/bold]: {company}  |  [bold]岗位[/bold]: {role}\n"
            f"[bold]输出目录[/bold]: {out}",
            title="jobagent · 面试备战计划生成",
            expand=False,
        )
    )

    app = build_graph()
    initial_state = {
        "job_input": job,
        "search_queries": [],
        "web_evidences": [],
        "kb_evidences": [],
        "all_evidences": [],
        "plan_json": None,
        "plan": None,
        "validation_errors": [],
        "retry_count": 0,
        "output_dir": out,
    }

    node_labels = {
        "intake": "解析 JD，提取检索词",
        "research": "Tavily 联网调研",
        "retrieve": "本地知识库召回",
        "rerank": "BGE Reranker 重排",
        "generate": "DeepSeek 生成备战计划",
        "validate": "校验 JSON 与引用",
        "render": "渲染并保存结果",
    }

    final_state: dict = {}
    with console.status("[bold green]运行中...[/bold green]") as status:
        for chunk in app.stream(initial_state, stream_mode="updates", config={"recursion_limit": 25}):
            for node_name, node_output in chunk.items():
                label = node_labels.get(node_name, node_name)
                status.update(f"[bold green]▶ {label}[/bold green]")
                console.log(f"  [dim]✓[/dim] {node_name} — {label}")
                if node_output is not None:
                    final_state.update(node_output)

    if not final_state.get("plan"):
        errors = final_state.get("validation_errors", [])
        console.print("\n[red bold]生成失败[/red bold] — 校验未通过：")
        for e in errors:
            console.print(f"  [red]• {e}[/red]")
        console.print(f"\n原始输出已保存至 [bold]{out}/plan.json[/bold]（如有）")
        sys.exit(1)

    plan_obj = final_state["plan"]
    console.print(f"\n[green bold]✓ 生成完成[/green bold] → {out}/")

    t = Table(title="计划摘要", show_header=True, header_style="bold cyan")
    t.add_column("章节", style="dim")
    t.add_column("条目数", justify="right")
    t.add_row("面试流程", str(len(plan_obj.interview_overview)))
    t.add_row("核心备考方向", str(len(plan_obj.focus_areas)))
    t.add_row("高频考点", str(len(plan_obj.high_frequency_topics)))
    t.add_row("Q&A 题库", str(len(plan_obj.q_and_a)))
    t.add_row("行动计划（周）", str(len(plan_obj.action_plan)))
    t.add_row("引用来源", str(len(plan_obj.citations)))
    console.print(t)
    console.print(f"\n[bold]plan.md[/bold]   → {out}/plan.md")
    console.print(f"[bold]plan.json[/bold] → {out}/plan.json")


if __name__ == "__main__":
    cli()
