from langgraph.graph import END, START, StateGraph

from src.agent.nodes import (
    intake_node,
    plan_node,
    rerank_node,
    render_node,
    research_node,
    retrieve_node,
    router_after_validate,
    validate_node,
)
from src.agent.state import AgentState
from src.schemas.plan import JobInput


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("intake", intake_node)
    g.add_node("research", research_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("rerank", rerank_node)
    g.add_node("generate", plan_node)   # "plan" conflicts with AgentState key; use "generate"
    g.add_node("validate", validate_node)
    g.add_node("render", render_node)

    g.add_edge(START, "intake")
    g.add_edge("intake", "research")
    g.add_edge("research", "retrieve")
    g.add_edge("retrieve", "rerank")
    g.add_edge("rerank", "generate")
    g.add_edge("generate", "validate")
    g.add_conditional_edges(
        "validate",
        router_after_validate,
        {"plan": "generate", "render": "render"},
    )
    g.add_edge("render", END)

    return g.compile()


def run_workflow(job_input: JobInput, output_dir: str = "runs/output") -> dict:
    """Run the full workflow and return the final state."""
    initial_state: AgentState = {
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
    app = build_graph()
    return app.invoke(initial_state, config={"recursion_limit": 25})
