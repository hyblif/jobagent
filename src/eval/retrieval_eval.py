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
BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
SCAN_N_CANDIDATES = (10, 20, 30)
SCAN_TOP_K = (3, 5, 8)


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
    query_prefix: str = "",
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
        query = f"{query_prefix}{case['query']}" if query_prefix else case["query"]
        for mode_name, use_rerank in modes.items():
            results = retrieve(
                query,
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


def scan_parameters(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run the small 2026-06-15 retrieval grid."""
    from src.rag.rerank import rerank

    runs = []
    max_candidates = max(SCAN_N_CANDIDATES)
    cached_cases: dict[bool, list[dict[str, Any]]] = {False: [], True: []}

    for use_query_prefix in (False, True):
        query_prefix = BGE_QUERY_PREFIX if use_query_prefix else ""
        for case in cases:
            query = f"{query_prefix}{case['query']}" if query_prefix else case["query"]
            vector_results = retrieve(
                query,
                n_candidates=max_candidates,
                top_k=max_candidates,
                use_rerank=False,
                raise_on_error=True,
            )
            reranked_results = rerank(query, vector_results, text_key="excerpt", top_k=max_candidates)
            cached_cases[use_query_prefix].append(
                {
                    "gold": case["gold"],
                    "vector_results": vector_results,
                    "reranked_results": reranked_results,
                }
            )

    for n_candidates in SCAN_N_CANDIDATES:
        for top_k in SCAN_TOP_K:
            for use_query_prefix in (False, True):
                no_rerank_scores = []
                rerank_scores = []
                for case_record in cached_cases[use_query_prefix]:
                    vector_subset = case_record["vector_results"][:n_candidates]
                    vector_ids = {item["id"] for item in vector_subset}
                    reranked_subset = [
                        item for item in case_record["reranked_results"] if item["id"] in vector_ids
                    ]
                    no_rerank_scores.append(score_results(vector_subset[:top_k], case_record["gold"]))
                    rerank_scores.append(score_results(reranked_subset[:top_k], case_record["gold"]))
                runs.append(
                    {
                        "config": {
                            "top_k": top_k,
                            "n_candidates": n_candidates,
                            "query_prefix": use_query_prefix,
                        },
                        "summary": {
                            "no_rerank": summarize(no_rerank_scores),
                            "rerank": summarize(rerank_scores),
                        },
                    }
                )
    return runs


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


def print_scan(runs: list[dict[str, Any]], total_cases: int) -> None:
    table = Table(title=f"Retrieval parameter scan ({total_cases} cases)")
    table.add_column("n_candidates", justify="right")
    table.add_column("top_k", justify="right")
    table.add_column("query_prefix")
    table.add_column("mode")
    table.add_column("hit@3", justify="right")
    table.add_column("hit@5", justify="right")
    table.add_column("MRR", justify="right")

    for run in runs:
        config = run["config"]
        for mode_name in ("no_rerank", "rerank"):
            metrics = run["summary"][mode_name]
            table.add_row(
                str(config["n_candidates"]),
                str(config["top_k"]),
                "yes" if config["query_prefix"] else "no",
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
    parser.add_argument("--query-prefix", default="", help="Optional query-side prefix before embedding/retrieval")
    parser.add_argument("--bge-query-prefix", action="store_true", help="Use the standard BGE retrieval query prefix")
    parser.add_argument("--scan", action="store_true", help="Run the 2026-06-15 parameter scan grid")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_eval_set(args.eval_set)
    query_prefix = BGE_QUERY_PREFIX if args.bge_query_prefix else args.query_prefix
    if args.scan and query_prefix:
        Console().print("[yellow]Warning: --query-prefix/--bge-query-prefix is ignored when --scan is active (scan always runs both prefix variants).[/yellow]")
    payload = (
        {"scan": scan_parameters(cases)}
        if args.scan
        else evaluate(cases, n_candidates=args.n_candidates, top_k=args.top_k, query_prefix=query_prefix)
    )
    output_path = args.out or Path("runs/eval") / f"{date.today().isoformat()}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "eval_set": str(args.eval_set),
        "run_date": date.today().isoformat(),
        "config": {
            "top_k": args.top_k,
            "n_candidates": args.n_candidates,
            "query_prefix": query_prefix or None,
            "scan": args.scan,
            "case_count": len(cases),
        },
        **payload,
    }
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.scan:
        print_scan(result["scan"], len(cases))
    else:
        print_summary(result["summary"], len(cases))
    Console().print(f"[green]Wrote eval results to {output_path}[/green]")


if __name__ == "__main__":
    main()
