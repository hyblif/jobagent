import os
from functools import lru_cache
from langchain_openai import ChatOpenAI


@lru_cache(maxsize=4)
def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not set. Copy .env.example to .env and fill it in."
        )
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        timeout=120,
        max_retries=2,
        model_kwargs={"response_format": {"type": "json_object"}},
    )
