"""
evaluation/benchmark.py
------------------------
Runs all three pipelines on every test question and records:

  Per pipeline, per question:
    - prompt_tokens, completion_tokens, total_tokens
    - cost_usd
    - latency_ms
    - answer text

  Then computes:
    - token reduction %  (GraphRAG vs Basic RAG)  ← 30% of judging score
    - cost reduction %   (GraphRAG vs Basic RAG)
    - latency comparison

Usage:
    python -m evaluation.benchmark \
        --questions data/test_questions.json \
        --output    data/benchmark_results.json
"""

from __future__ import annotations

import json
import argparse
import statistics
from typing import Any

from tqdm import tqdm
import pipelines.llm_only  as p1
import pipelines.basic_rag as p2
import pipelines.graphrag  as p3


def run_benchmark(
    questions: list[dict],
    persist_dir: str = "./data/chroma_db",
) -> dict[str, Any]:
    """
    Run all three pipelines on every question.

    Args:
        questions: list of {"question": str, "ground_truth": str}
        persist_dir: ChromaDB directory for Basic RAG

    Returns:
        Full benchmark results dict (saved to JSON, fed to dashboard + accuracy eval).
    """
    rows = []

    for item in tqdm(questions, desc="Benchmarking"):
        q  = item["question"]
        gt = item.get("ground_truth", "")

        r1 = p1.run(q)
        r2 = p2.run(q, persist_dir=persist_dir)
        r3 = p3.run(q)

        rows.append({
            "question":     q,
            "ground_truth": gt,
            "llm_only":     {k: v for k, v in r1.items() if k not in ("context_chunks",)},
            "basic_rag":    {k: v for k, v in r2.items() if k not in ("context_chunks",)},
            "graphrag":     {k: v for k, v in r3.items() if k not in ("context_chunks", "service_response")},
        })

    return _compute_summary(rows)


def _compute_summary(rows: list[dict]) -> dict[str, Any]:
    def avg(key: str, pipeline: str) -> float:
        vals = [r[pipeline].get(key, 0) for r in rows]
        return round(statistics.mean(vals), 4) if vals else 0.0

    p1_tokens  = avg("total_tokens", "llm_only")
    p2_tokens  = avg("total_tokens", "basic_rag")
    p3_tokens  = avg("total_tokens", "graphrag")

    p1_cost    = avg("cost_usd",   "llm_only")
    p2_cost    = avg("cost_usd",   "basic_rag")
    p3_cost    = avg("cost_usd",   "graphrag")

    p1_lat     = avg("latency_ms", "llm_only")
    p2_lat     = avg("latency_ms", "basic_rag")
    p3_lat     = avg("latency_ms", "graphrag")

    # Token reduction: GraphRAG vs Basic RAG (the headline metric)
    token_reduction_vs_rag = (
        round((p2_tokens - p3_tokens) / p2_tokens * 100, 2) if p2_tokens else 0
    )
    cost_reduction_vs_rag = (
        round((p2_cost - p3_cost) / p2_cost * 100, 2) if p2_cost else 0
    )

    return {
        "n_questions": len(rows),
        "summary": {
            "llm_only":  {"avg_total_tokens": p1_tokens, "avg_cost_usd": p1_cost, "avg_latency_ms": p1_lat},
            "basic_rag": {"avg_total_tokens": p2_tokens, "avg_cost_usd": p2_cost, "avg_latency_ms": p2_lat},
            "graphrag":  {"avg_total_tokens": p3_tokens, "avg_cost_usd": p3_cost, "avg_latency_ms": p3_lat},
            "token_reduction_graphrag_vs_rag_pct": token_reduction_vs_rag,
            "cost_reduction_graphrag_vs_rag_pct":  cost_reduction_vs_rag,
        },
        "questions": rows,
    }


def print_summary(results: dict) -> None:
    s = results["summary"]
    print("\n" + "=" * 70)
    print(f"{'Pipeline':<14} {'Avg Tokens':>12} {'Avg Cost':>12} {'Avg Latency':>14}")
    print("-" * 70)
    for key, label in [("llm_only","LLM-Only"), ("basic_rag","Basic RAG"), ("graphrag","GraphRAG")]:
        d = s[key]
        print(
            f"{label:<14}"
            f"{d['avg_total_tokens']:>12.0f}"
            f"  ${d['avg_cost_usd']:>10.6f}"
            f"  {d['avg_latency_ms']:>10.0f}ms"
        )
    print("=" * 70)
    tr = s["token_reduction_graphrag_vs_rag_pct"]
    cr = s["cost_reduction_graphrag_vs_rag_pct"]
    arrow = "▼" if tr > 0 else "▲"
    print(f"\nGraphRAG vs Basic RAG:")
    print(f"  Token reduction : {arrow} {abs(tr):.1f}%")
    print(f"  Cost reduction  : {arrow} {abs(cr):.1f}%")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions",   default="data/test_questions.json")
    parser.add_argument("--output",      default="data/benchmark_results.json")
    parser.add_argument("--persist-dir", default="./data/chroma_db")
    args = parser.parse_args()

    with open(args.questions) as f:
        questions = json.load(f)

    print(f"Running benchmark on {len(questions)} questions...")
    results = run_benchmark(questions, persist_dir=args.persist_dir)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print_summary(results)
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()