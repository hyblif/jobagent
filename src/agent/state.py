from typing import TypedDict

from src.schemas.plan import Evidence, JobInput, PrepPlan


class AgentState(TypedDict):
    job_input: JobInput
    search_queries: list[str]
    web_evidences: list[Evidence]
    kb_evidences: list[Evidence]
    all_evidences: list[Evidence]
    plan_json: dict | None
    plan: PrepPlan | None
    validation_errors: list[str]
    retry_count: int
    output_dir: str
