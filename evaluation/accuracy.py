"""
evaluation/accuracy.py
-----------------------
Answer accuracy evaluation using two complementary HuggingFace approaches:

1. LLM-as-a-Judge
   A free hosted HuggingFace model grades each answer PASS/FAIL against
   the reference answer. Bonus threshold: ≥ 90% pass rate.

2. BERTScore
   Measures semantic similarity between generated and reference answers.
   Bonus thresholds:
     - BERTScore F1 rescaled ≥ 0.55
     - BERTScore F1 raw      ≥ 0.88

Usage:
    python -m evaluation.accuracy \
        --results  data/benchmark_results.json \
        --output   data/accuracy_report.json
"""

from __future__ import annotations

import json
import argparse
from typing import Any

from bert_score import score as bert_score
from transformers import pipeline as hf_pipeline


# ── LLM-as-a-Judge ───────────────────────────────────────────────────────────

JUDGE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"   # free HF hosted model

_judge_pipeline = None

def _get_judge():
    global _judge_pipeline
    if _judge_pipeline is None:
        print(f"Loading judge model ({JUDGE_MODEL})...")
        _judge_pipeline = hf_pipeline(
            "text-generation",
            model=JUDGE_MODEL,
            device_map="auto",
            max_new_tokens=64,
        )
    return _judge_pipeline

JUDGE_PROMPT = """\
You are an answer quality judge. Given a QUESTION, a REFERENCE ANSWER, and a CANDIDATE ANSWER,
decide if the candidate answer is correct and complete relative to the reference.

Respond with exactly one word: PASS or FAIL.

QUESTION: {question}
REFERENCE ANSWER: {reference}
CANDIDATE ANSWER: {candidate}

Verdict:"""


def llm_judge(question: str, reference: str, candidate: str) -> str:
    """
    Returns "PASS" or "FAIL".
    """
    judge = _get_judge()
    prompt = JUDGE_PROMPT.format(
        question=question, reference=reference, candidate=candidate
    )
    out = judge(prompt, return_full_text=False)[0]["generated_text"].strip().upper()
    return "PASS" if "PASS" in out else "FAIL"


def batch_judge(
    questions: list[str],
    references: list[str],
    candidates: list[str],
) -> list[str]:
    return [
        llm_judge(q, r, c)
        for q, r, c in zip(questions, references, candidates)
    ]


# ── BERTScore ─────────────────────────────────────────────────────────────────

BERT_MODEL = "microsoft/deberta-xlarge-mnli"   # recommended by bert-score docs

def compute_bertscore(
    candidates: list[str],
    references: list[str],
    model: str = BERT_MODEL,
    lang: str = "en",
) -> dict[str, list[float]]:
    """
    Compute BERTScore P, R, F1 (raw and rescaled).

    Returns:
        {
          "precision_raw": [...],
          "recall_raw": [...],
          "f1_raw": [...],
          "f1_rescaled": [...],   ← bonus threshold: ≥ 0.55
        }
    """
    P, R, F1 = bert_score(
        candidates, references,
        model_type=model,
        lang=lang,
        rescale_with_baseline=False,
        verbose=True,
    )
    P_r, R_r, F1_r = bert_score(
        candidates, references,
        model_type=model,
        lang=lang,
        rescale_with_baseline=True,
        verbose=False,
    )
    return {
        "precision_raw": [round(x.item(), 4) for x in P],
        "recall_raw":    [round(x.item(), 4) for x in R],
        "f1_raw":        [round(x.item(), 4) for x in F1],
        "f1_rescaled":   [round(x.item(), 4) for x in F1_r],
    }


# ── Combined evaluation ───────────────────────────────────────────────────────

def evaluate_pipeline(
    pipeline_name: str,
    questions: list[str],
    references: list[str],
    candidates: list[str],
    skip_judge: bool = False,
) -> dict[str, Any]:
    """
    Run both evaluations for one pipeline.

    Returns a dict with per-question results and aggregate stats.
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {pipeline_name}")
    print(f"{'='*60}")

    # BERTScore
    print("Running BERTScore...")
    bs = compute_bertscore(candidates, references)

    # LLM-as-a-Judge
    verdicts = []
    if not skip_judge:
        print("Running LLM-as-a-Judge...")
        verdicts = batch_judge(questions, references, candidates)
    else:
        verdicts = ["SKIP"] * len(questions)

    # Per-question rows
    rows = []
    for i, (q, ref, cand, verdict) in enumerate(
        zip(questions, references, candidates, verdicts)
    ):
        rows.append({
            "question":       q,
            "reference":      ref,
            "candidate":      cand,
            "judge_verdict":  verdict,
            "bertscore_f1_raw":       bs["f1_raw"][i],
            "bertscore_f1_rescaled":  bs["f1_rescaled"][i],
            "bertscore_precision":    bs["precision_raw"][i],
            "bertscore_recall":       bs["recall_raw"][i],
        })

    # Aggregate
    n = len(rows)
    pass_count    = sum(1 for r in rows if r["judge_verdict"] == "PASS")
    pass_rate     = round(pass_count / n, 4) if n else 0
    avg_f1_raw    = round(sum(bs["f1_raw"]) / n, 4) if n else 0
    avg_f1_scaled = round(sum(bs["f1_rescaled"]) / n, 4) if n else 0

    # Bonus thresholds (from problem statement)
    bonus_judge    = pass_rate >= 0.90
    bonus_bert_r   = avg_f1_rescaled >= 0.55
    bonus_bert_raw = avg_f1_raw >= 0.88
    bonus_max      = bonus_bert_r and bonus_judge

    agg = {
        "pipeline":              pipeline_name,
        "n_questions":           n,
        "judge_pass_count":      pass_count,
        "judge_pass_rate":       pass_rate,
        "bertscore_f1_raw_avg":  avg_f1_raw,
        "bertscore_f1_rescaled_avg": avg_f1_scaled,
        # Bonus flags
        "bonus_judge_gte90":     bonus_judge,
        "bonus_bert_rescaled_gte055": bonus_bert_r,
        "bonus_bert_raw_gte088": bonus_bert_raw,
        "bonus_max_unlocked":    bonus_max,
    }

    print(f"\nResults for {pipeline_name}:")
    print(f"  LLM-as-a-Judge pass rate : {pass_rate:.1%}  {'✅ BONUS' if bonus_judge else ''}")
    print(f"  BERTScore F1 (rescaled)  : {avg_f1_scaled:.4f}  {'✅ BONUS' if bonus_bert_r else ''}")
    print(f"  BERTScore F1 (raw)       : {avg_f1_raw:.4f}  {'✅ BONUS' if bonus_bert_raw else ''}")
    if bonus_max:
        print("  🏆 MAXIMUM BONUS UNLOCKED (both thresholds hit)")

    return {"aggregate": agg, "per_question": rows}


def run_full_evaluation(
    benchmark_results: dict,
    skip_judge: bool = False,
) -> dict[str, Any]:
    """
    Run accuracy evaluation for all three pipelines from benchmark results.

    Args:
        benchmark_results: Output from evaluation/benchmark.py
        skip_judge: Skip LLM-as-a-Judge (faster, for testing)
    """
    questions  = [r["question"]      for r in benchmark_results["questions"]]
    references = [r["ground_truth"]  for r in benchmark_results["questions"]]

    pipeline_results = {}
    for pipeline_key in ["llm_only", "basic_rag", "graphrag"]:
        candidates = [
            r[pipeline_key]["answer"] for r in benchmark_results["questions"]
        ]
        label = {"llm_only": "LLM-Only", "basic_rag": "Basic RAG", "graphrag": "GraphRAG"}[pipeline_key]
        pipeline_results[pipeline_key] = evaluate_pipeline(
            pipeline_name=label,
            questions=questions,
            references=references,
            candidates=candidates,
            skip_judge=skip_judge,
        )

    return pipeline_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results",    default="data/benchmark_results.json")
    parser.add_argument("--output",     default="data/accuracy_report.json")
    parser.add_argument("--skip-judge", action="store_true", help="Skip LLM-as-a-Judge (faster)")
    args = parser.parse_args()

    with open(args.results) as f:
        benchmark_results = json.load(f)

    report = run_full_evaluation(benchmark_results, skip_judge=args.skip_judge)

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nAccuracy report saved to: {args.output}")


if __name__ == "__main__":
    main()