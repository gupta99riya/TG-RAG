"""
pipelines/graphrag.py
----------------------
Pipeline 3: GraphRAG via TigerGraph GraphRAG REST API

This wraps the official TigerGraph GraphRAG service
(github.com/tigergraph/graphrag) deployed via Docker.

The GraphRAG service handles:
  - Knowledge graph construction from your corpus
  - Hybrid retrieval (vector search + multi-hop graph traversal)
  - Context assembly

We call its /query endpoint, capture the response + token metadata,
and return the standard dashboard dict.

GraphRAG service defaults to http://localhost:80 (nginx proxy).
Override via env var: GRAPHRAG_BASE_URL
"""

from __future__ import annotations
import os
import time
import tiktoken
from typing import Any

import requests

from pipelines.llm_client import _count_tokens, _cost, default_client

# ── Config ────────────────────────────────────────────────────────────────────

GRAPHRAG_BASE_URL = os.getenv("GRAPHRAG_BASE_URL", "http://localhost:80")
GRAPHRAG_GRAPH    = os.getenv("GRAPHRAG_GRAPH", "MyGraph")
GRAPHRAG_USER     = os.getenv("GRAPHRAG_USER", "tigergraph")
GRAPHRAG_PASS     = os.getenv("GRAPHRAG_PASS", "tigergraph")

# Retriever to use: "hybrid" (default), "community", "sibling"
GRAPHRAG_RETRIEVER = os.getenv("GRAPHRAG_RETRIEVER", "hybrid")

# Tunable retrieval parameters (match graphrag_config in server_config.json)
TOP_K      = int(os.getenv("GRAPHRAG_TOP_K", "5"))
NUM_HOPS   = int(os.getenv("GRAPHRAG_NUM_HOPS", "2"))


# ── Session + auth ────────────────────────────────────────────────────────────

_session: requests.Session | None = None

def _get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
        _session.auth = (GRAPHRAG_USER, GRAPHRAG_PASS)
        _session.headers.update({"Content-Type": "application/json"})
    return _session


# ── API calls ─────────────────────────────────────────────────────────────────

def query_graphrag(
    question: str,
    graph: str = GRAPHRAG_GRAPH,
    retriever: str = GRAPHRAG_RETRIEVER,
    top_k: int = TOP_K,
    num_hops: int = NUM_HOPS,
) -> dict:
    """
    Call the TigerGraph GraphRAG /query endpoint.

    Returns the raw JSON response from the service.
    """
    session = _get_session()
    url = f"{GRAPHRAG_BASE_URL}/api/query"

    payload = {
        "graph_name": graph,
        "query":      question,
        "method":     retriever,
        "top_k":      top_k,
        "num_hops":   num_hops,
    }

    resp = session.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def ingest_documents_to_graphrag(
    file_paths: list[str],
    graph: str = GRAPHRAG_GRAPH,
) -> dict:
    """
    Upload local files to the GraphRAG service for ingestion.

    Args:
        file_paths: Absolute paths to local PDF/TXT/MD files.
        graph:      Graph name to ingest into.
    """
    session = _get_session()
    url = f"{GRAPHRAG_BASE_URL}/api/documents/upload"

    files = [
        ("files", (os.path.basename(p), open(p, "rb"), "application/octet-stream"))
        for p in file_paths
    ]
    data = {"graph_name": graph}

    resp = session.post(url, files=files, data=data, timeout=300)
    resp.raise_for_status()
    return resp.json()


def init_knowledge_graph(graph: str = GRAPHRAG_GRAPH) -> dict:
    """
    Initialize the TigerGraph schema and install queries for a new graph.
    Run this once before ingestion.
    """
    session = _get_session()
    url = f"{GRAPHRAG_BASE_URL}/api/graphrag/init"
    resp = session.post(url, json={"graph_name": graph}, timeout=300)
    resp.raise_for_status()
    return resp.json()


def rebuild_knowledge_graph(graph: str = GRAPHRAG_GRAPH) -> dict:
    """
    Trigger a knowledge graph rebuild after ingestion.
    Runs entity extraction + community detection.
    """
    session = _get_session()
    url = f"{GRAPHRAG_BASE_URL}/api/graphrag/rebuild"
    resp = session.post(url, json={"graph_name": graph}, timeout=600)
    resp.raise_for_status()
    return resp.json()


# ── Token accounting ──────────────────────────────────────────────────────────

def _estimate_tokens_from_response(api_resp: dict, question: str) -> tuple[int, int]:
    """
    Extract or estimate prompt/completion tokens from the GraphRAG API response.

    The GraphRAG service may or may not return token usage.
    If not, we estimate from the retrieved context + answer.
    """
    # Try to read from response metadata
    usage = api_resp.get("usage") or api_resp.get("token_usage") or {}
    if usage.get("prompt_tokens"):
        return usage["prompt_tokens"], usage.get("completion_tokens", 0)

    # Estimate: context text + question = prompt; answer = completion
    context_text = " ".join(
        chunk.get("text", "") or chunk.get("content", "")
        for chunk in api_resp.get("context", [])
    )
    prompt_est     = _count_tokens(question + context_text)
    completion_est = _count_tokens(api_resp.get("result", "") or api_resp.get("answer", ""))
    return prompt_est, completion_est


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(
    query: str,
    graph: str = GRAPHRAG_GRAPH,
    retriever: str = GRAPHRAG_RETRIEVER,
    top_k: int = TOP_K,
    num_hops: int = NUM_HOPS,
) -> dict[str, Any]:
    """
    Run the GraphRAG pipeline via TigerGraph GraphRAG REST API.

    Returns the standard dashboard dict including token counts and cost.
    Falls back to Basic-RAG-style local inference if the service is unreachable.
    """
    t0 = time.time()
    try:
        api_resp = query_graphrag(
            question=query,
            graph=graph,
            retriever=retriever,
            top_k=top_k,
            num_hops=num_hops,
        )
        ms = round((time.time() - t0) * 1000)

        # Extract answer
        answer = (
            api_resp.get("result")
            or api_resp.get("answer")
            or api_resp.get("response")
            or str(api_resp)
        )

        # Extract retrieved context chunks
        context_chunks = api_resp.get("context") or api_resp.get("chunks") or []

        # Token accounting
        pt, ct = _estimate_tokens_from_response(api_resp, query)

        return {
            "pipeline":          "GraphRAG",
            "answer":            answer,
            "prompt_tokens":     pt,
            "completion_tokens": ct,
            "total_tokens":      pt + ct,
            "cost_usd":          _cost(default_client.model, pt, ct),
            "latency_ms":        ms,
            "model":             default_client.model,
            "context_chunks":    context_chunks,
            "context_text":      "\n\n".join(
                c.get("text", "") or c.get("content", "") for c in context_chunks
            ),
            "graph":             graph,
            "retriever":         retriever,
            "num_hops":          num_hops,
            "service_response":  api_resp,
        }

    except (requests.ConnectionError, requests.Timeout) as e:
        # GraphRAG service not running — return an error result for the dashboard
        ms = round((time.time() - t0) * 1000)
        return {
            "pipeline":          "GraphRAG",
            "answer":            f"[GraphRAG service unavailable: {e}]",
            "prompt_tokens":     0,
            "completion_tokens": 0,
            "total_tokens":      0,
            "cost_usd":          0.0,
            "latency_ms":        ms,
            "model":             default_client.model,
            "context_chunks":    [],
            "context_text":      "",
            "error":             str(e),
        }


if __name__ == "__main__":
    r = run("What is GraphRAG and how does it differ from standard RAG?")
    print(f"[{r['pipeline']}] {r['total_tokens']} tokens | ${r['cost_usd']:.6f} | {r['latency_ms']}ms")
    print(r["answer"])