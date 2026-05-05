"""
dashboard/app.py
-----------------
Interactive comparison dashboard built with Gradio.

One query in → all 3 pipelines run → side-by-side responses + metrics out.

Metrics displayed per pipeline:
  - Tokens used (prompt + completion)
  - Response latency (ms)
  - Cost per query (USD)
  - Answer accuracy (LLM-as-a-Judge PASS/FAIL + BERTScore)
  - Token reduction % vs Basic RAG  ← headline metric

Run with:
    python -m dashboard.app
or via the CLI entry-point:
    graphrag-dashboard
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

import pipelines.llm_only  as p1
import pipelines.basic_rag as p2
import pipelines.graphrag  as p3
from pipelines.llm_client import get_client, default_client as _dc

# ── LLM-as-a-judge (lightweight inline version for dashboard) ─────────────────

_judge = None

def _load_judge():
    global _judge
    if _judge is None:
        from transformers import pipeline as hf_pipeline
        _judge = hf_pipeline(
            "text-generation",
            model="HuggingFaceH4/zephyr-7b-beta",
            device_map="auto",
            max_new_tokens=10,
        )
    return _judge


JUDGE_PROMPT = (
    "Given the QUESTION and REFERENCE ANSWER, is the CANDIDATE ANSWER correct?\n"
    "Reply with exactly PASS or FAIL.\n\n"
    "QUESTION: {q}\nREFERENCE: {ref}\nCANDIDATE: {cand}\n\nVerdict:"
)


def _judge_answer(question: str, reference: str, candidate: str) -> str:
    if not reference.strip():
        return "N/A (no reference)"
    try:
        judge = _load_judge()
        prompt = JUDGE_PROMPT.format(q=question, ref=reference, cand=candidate)
        out = judge(prompt, return_full_text=False)[0]["generated_text"].strip().upper()
        return "✅ PASS" if "PASS" in out else "❌ FAIL"
    except Exception as e:
        return f"Judge error: {e}"


def _bertscore(reference: str, candidate: str) -> str:
    if not reference.strip():
        return "N/A"
    try:
        from bert_score import score as bs
        _, _, F1 = bs([candidate], [reference], lang="en", verbose=False)
        raw = F1[0].item()
        _, _, F1r = bs([candidate], [reference], lang="en",
                       rescale_with_baseline=True, verbose=False)
        rescaled = F1r[0].item()
        r_tag  = " ✅" if rescaled >= 0.55 else ""
        raw_tag = " ✅" if raw >= 0.88 else ""
        return f"F1 rescaled: {rescaled:.3f}{r_tag} | F1 raw: {raw:.3f}{raw_tag}"
    except Exception as e:
        return f"BERTScore error: {e}"


# ── Core query runner ─────────────────────────────────────────────────────────

def run_all(
    query: str,
    reference_answer: str,
    rag_top_k: int,
    graphrag_num_hops: int,
    graphrag_retriever: str,
    run_accuracy: bool,
) -> tuple:
    """
    Called by Gradio when the user clicks Submit.
    Returns all UI output components.
    """
    if not query.strip():
        empty = "Please enter a query."
        return (empty,) * 3 + ("",) * 3 + (pd.DataFrame(),) + ("",) * 6

    # Run pipelines
    r1 = p1.run(query)
    r2 = p2.run(query, top_k=rag_top_k)
    r3 = p3.run(query, num_hops=graphrag_num_hops, retriever=graphrag_retriever)

    # Answers
    ans1, ans2, ans3 = r1["answer"], r2["answer"], r3["answer"]

    # Token reduction (vs Basic RAG — the headline metric)
    p2_tok = r2["total_tokens"] or 1
    p3_tok = r3["total_tokens"]
    reduction_pct = round((p2_tok - p3_tok) / p2_tok * 100, 1)
    reduction_label = (
        f"{'▼' if reduction_pct >= 0 else '▲'} {abs(reduction_pct):.1f}% vs Basic RAG"
    )

    # Metrics table
    metrics_df = pd.DataFrame([
        {
            "Pipeline":        "🤖 LLM-Only",
            "Prompt Tokens":   r1["prompt_tokens"],
            "Completion Tokens": r1["completion_tokens"],
            "Total Tokens":    r1["total_tokens"],
            "Cost (USD)":      f"${r1['cost_usd']:.6f}",
            "Latency (ms)":    r1["latency_ms"],
            "Token Δ vs RAG":  f"{round((p2_tok - r1['total_tokens'])/p2_tok*100,1):+.1f}%",
        },
        {
            "Pipeline":        "📚 Basic RAG",
            "Prompt Tokens":   r2["prompt_tokens"],
            "Completion Tokens": r2["completion_tokens"],
            "Total Tokens":    r2["total_tokens"],
            "Cost (USD)":      f"${r2['cost_usd']:.6f}",
            "Latency (ms)":    r2["latency_ms"],
            "Token Δ vs RAG":  "baseline",
        },
        {
            "Pipeline":        "🐯 GraphRAG",
            "Prompt Tokens":   r3["prompt_tokens"],
            "Completion Tokens": r3["completion_tokens"],
            "Total Tokens":    r3["total_tokens"],
            "Cost (USD)":      f"${r3['cost_usd']:.6f}",
            "Latency (ms)":    r3["latency_ms"],
            "Token Δ vs RAG":  reduction_label,
        },
    ])

    # Accuracy (optional — loads heavy models)
    acc1 = acc2 = acc3 = ""
    if run_accuracy and reference_answer.strip():
        acc1 = f"Judge: {_judge_answer(query, reference_answer, ans1)}\nBERT: {_bertscore(reference_answer, ans1)}"
        acc2 = f"Judge: {_judge_answer(query, reference_answer, ans2)}\nBERT: {_bertscore(reference_answer, ans2)}"
        acc3 = f"Judge: {_judge_answer(query, reference_answer, ans3)}\nBERT: {_bertscore(reference_answer, ans3)}"

    return ans1, ans2, ans3, acc1, acc2, acc3, metrics_df, reduction_label


# ── Gradio UI ─────────────────────────────────────────────────────────────────

CSS = """
.pipeline-header { font-size: 1.1em; font-weight: bold; }
.metric-positive { color: #16a34a; font-weight: bold; }
.metric-negative { color: #dc2626; }
footer { display: none !important; }
"""

DESCRIPTION = """
# 🐯 GraphRAG Inference Hackathon — Comparison Dashboard

Enter a query to run it through **all three pipelines simultaneously**:
- **LLM-Only** — no retrieval, parametric memory only
- **Basic RAG** — vector similarity search (ChromaDB)
- **GraphRAG** — TigerGraph multi-hop graph traversal + hybrid retrieval

**Headline metric:** Token reduction with maintained accuracy.
"""

def launch():
    with gr.Blocks(css=CSS, title="GraphRAG Dashboard") as demo:
        gr.Markdown(DESCRIPTION)

        # ── Inputs ────────────────────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=3):
                query_box = gr.Textbox(
                    label="Query",
                    placeholder="Ask a question about your dataset...",
                    lines=3,
                )
                ref_box = gr.Textbox(
                    label="Reference Answer (optional — enables accuracy evaluation)",
                    placeholder="Paste the expected answer for LLM-as-a-Judge + BERTScore evaluation",
                    lines=2,
                )

            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Parameters")
                top_k_slider = gr.Slider(
                    minimum=1, maximum=15, value=5, step=1,
                    label="Basic RAG top_k",
                )
                hops_slider = gr.Slider(
                    minimum=1, maximum=4, value=2, step=1,
                    label="GraphRAG num_hops",
                )
                retriever_dd = gr.Dropdown(
                    choices=["hybrid", "community", "sibling"],
                    value="hybrid",
                    label="GraphRAG Retriever",
                )
                accuracy_toggle = gr.Checkbox(
                    label="Run accuracy evaluation (slower)",
                    value=False,
                )
                submit_btn = gr.Button("▶ Run All Pipelines", variant="primary")

        # ── Metrics table ─────────────────────────────────────────────────────
        with gr.Row():
            metrics_table = gr.Dataframe(
                label="📊 Token / Cost / Latency Comparison",
                wrap=True,
            )

        reduction_label = gr.Markdown("")

        # ── Side-by-side answers ──────────────────────────────────────────────
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🤖 Pipeline 1 — LLM-Only")
                ans1 = gr.Textbox(label="Answer", lines=10, interactive=False)
                acc1 = gr.Textbox(label="Accuracy", lines=2, interactive=False)

            with gr.Column():
                gr.Markdown("### 📚 Pipeline 2 — Basic RAG")
                ans2 = gr.Textbox(label="Answer", lines=10, interactive=False)
                acc2 = gr.Textbox(label="Accuracy", lines=2, interactive=False)

            with gr.Column():
                gr.Markdown("### 🐯 Pipeline 3 — GraphRAG")
                ans3 = gr.Textbox(label="Answer", lines=10, interactive=False)
                acc3 = gr.Textbox(label="Accuracy", lines=2, interactive=False)

        # ── Example queries ───────────────────────────────────────────────────
        gr.Examples(
            examples=[
                ["What is the main topic of the dataset?", ""],
                ["How are the key entities related to each other?", ""],
                ["Summarize the most important findings.", ""],
            ],
            inputs=[query_box, ref_box],
        )

        # ── Wire up ───────────────────────────────────────────────────────────
        submit_btn.click(
            fn=run_all,
            inputs=[
                query_box, ref_box,
                top_k_slider, hops_slider, retriever_dd,
                accuracy_toggle,
            ],
            outputs=[ans1, ans2, ans3, acc1, acc2, acc3, metrics_table, reduction_label],
        )

    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    launch()