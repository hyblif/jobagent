"""Evaluate retrieval quality against a fixed source/heading gold set."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from src.rag.retriever import retrieve

DEFAULT_EVAL_SET = Path(__file__).with_name("eval_set.json")
DEFAULT_TOP_K = 5
DEFAULT_N_CANDIDATES = 20


def load_eval_set(path: Path = DEFAULT_EVAL_SET) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases", [])
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"No eval cases found in {path}")
    return cases


def matches_gold(result: dict[str, Any], gold: list[dict[str, str]]) -> bool:
    source = result.get("source") or result.get("url_or_path")
    heading = result.get("heading")
    return any(source == item.get("source") and heading == item.get("heading") for item in gold)


def score_results(results: list[dict[str, Any]], gold: list[dict[str, str]]) -> dict[str, Any]:
    first_hit_rank = None
    for idx, result in enumerate(results, start=1):
        if matches_gold(result, gold):
            first_hit_rank = idx
            break

    return {
        "rank": first_hit_rank,
        "hit@3": first_hit_rank is not None and first_hit_rank <= 3,
        "hit@5": first_hit_rank is not None and first_hit_rank <= 5,
        "mrr": 0.0 if first_hit_rank is None else 1.0 / first_hit_rank,
    }


def summarize(scores: list[dict[str, Any]]) -> dict[str, float]:
    total = len(scores)
    if total == 0:
        return {"hit@3": 0.0, "hit@5": 0.0, "mrr": 0.0}
    return {
        "hit@3": sum(1 for score in scores if score["hit@3"]) / total,
        "hit@5": sum(1 for score in scores if score["hit@5"]) / total,
        "mrr": sum(float(score["mrr"]) for score in scores) / total,
    }


def _compact_result(result: dict[str, Any], rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "id": result.get("id"),
        "source": result.get("source") or result.get("url_or_path"),
        "heading": result.get("heading"),
        "title": result.get("title"),
        "score": result.get("score"),
    }


def evaluate(
    cases: list[dict[str, Any]],
    *,
    n_candidates: int = DEFAULT_N_CANDIDATES,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    modes = {
        "no_rerank": False,
        "rerank": True,
    }
    per_case: list[dict[str, Any]] = []
    scores_by_mode: dict[str, list[dict[str, Any]]] = {name: [] for name in modes}

    for case in cases:
        case_record: dict[str, Any] = {
            "id": case["id"],
            "kind": case.get("kind"),
            "query": case["query"],
            "gold": case["gold"],
            "modes": {},
        }
        for mode_name, use_rerank in modes.items():
            results = retrieve(
                case["query"],
                n_candidates=n_candidates,
                top_k=top_k,
                use_rerank=use_rerank,
                raise_on_error=True,
            )
            score = score_results(results, case["gold"])
            scores_by_mode[mode_name].append(score)
            case_record["modes"][mode_name] = {
                "score": score,
                "results": [_compact_result(result, rank) for rank, result in enumerate(results, start=1)],
            }
        per_case.append(case_record)

    return {
        "summary": {mode_name: summarize(scores) for mode_name, scores in scores_by_mode.items()},
        "cases": per_case,
    }


def print_summary(summary: dict[str, dict[str, float]], total_cases: int) -> None:
    table = Table(title=f"Retrieval eval ({total_cases} cases)")
    table.add_column("Mode")
    table.add_column("hit@3", justify="right")
    table.add_column("hit@5", justify="right")
    table.add_column("MRR", justify="right")

    for mode_name in ("no_rerank", "rerank"):
        metrics = summary[mode_name]
        table.add_row(
            mode_name,
            f"{metrics['hit@3']:.3f}",
            f"{metrics['hit@5']:.3f}",
            f"{metrics['mrr']:.3f}",
        )

    Console().print(table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Chroma retrieval with and without reranker")
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--n-candidates", type=int, default=DEFAULT_N_CANDIDATES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_eval_set(args.eval_set)
    payload = evaluate(cases, n_candidates=args.n_candidates, top_k=args.top_k)
    output_path = args.out or Path("runs/eval") / f"{date.today().isoformat()}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "eval_set": str(args.eval_set),
        "run_date": date.today().isoformat(),
        "config": {
            "top_k": args.top_k,
            "n_candidates": args.n_candidates,
            "case_count": len(cases),
        },
        **payload,
    }
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print_summary(result["summary"], len(cases))
    Console().print(f"[green]Wrote eval results to {output_path}[/green]")


if __name__ == "__main__":
    main()
