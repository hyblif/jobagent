from typing import Literal
from pydantic import BaseModel


class JobInput(BaseModel):
    company: str
    role: str
    jd_text: str


class Evidence(BaseModel):
    id: str  # "web-{n}" or "kb-{n}", assigned exclusively in rerank_node
    source_type: Literal["web", "kb"]
    title: str
    url_or_path: str
    excerpt: str
    score: float | None = None


class PrepPlan(BaseModel):
    interview_overview: list[dict]    # [{phase, description, source_ids}]
    focus_areas: list[dict]           # [{area, importance, source_ids}]
    high_frequency_topics: list[dict] # [{topic, source_ids}]
    q_and_a: list[dict]               # [{question, answer_outline, source_ids}]
    action_plan: list[dict]           # [{week, tasks, source_ids}]
    citations: list[dict]             # [{id, title, url_or_path, excerpt}]
