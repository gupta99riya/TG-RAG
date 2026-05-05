# GraphRAG Inference Hackathon — Complete Project

> **Headline metric:** Token reduction with maintained accuracy.
> Show how GraphRAG cuts tokens vs Basic RAG without dropping answer quality.

---

## What This Project Does

Runs the same query through three pipelines simultaneously and compares:

| Metric | Weight |
|---|---|
| **Token reduction** (GraphRAG vs Basic RAG) | 30% |
| **Answer accuracy** (LLM-as-a-Judge + BERTScore) | 30% |
| **Performance** (latency, throughput) | 20% |
| **Engineering & Storytelling** | 20% |

**Bonus points:** LLM-as-a-Judge pass rate ≥ 90% AND BERTScore F1 rescaled ≥ 0.55

---

## Architecture

```
                         User Query
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
   ┌──────▼──────┐   ┌───────▼──────┐   ┌──────▼──────────┐
   │ LLM-Only    │   │  Basic RAG   │   │    GraphRAG      │
   │             │   │              │   │                  │
   │  No context │   │  ChromaDB    │   │  TigerGraph      │
   │  Pure LLM   │   │  vector search│  │  GraphRAG Repo   │
   │  parametric │   │  top-k chunks│   │  hybrid search   │
   │  memory     │   │  + LLM       │   │  (vector+graph)  │
   └──────┬──────┘   └───────┬──────┘   └──────┬──────────┘
          │                  │                  │
          └──────────────────┼──────────────────┘
                             │
                    ┌────────▼────────┐
                    │ Dashboard       │
                    │ Side-by-side    │
                    │ tokens/cost/    │
                    │ latency/accuracy│
                    └─────────────────┘
```

---

## Project Structure

```
graphrag_hackathon/
├── pipelines/
│   ├── llm_client.py     ← vendor-agnostic LLM abstraction (OpenAI/Claude/Groq/Gemini)
│   │                        with token counting + cost calculation
│   ├── llm_only.py       ← Pipeline 1: no retrieval
│   ├── basic_rag.py      ← Pipeline 2: ChromaDB vector search
│   └── graphrag.py       ← Pipeline 3: TigerGraph GraphRAG REST API wrapper
├── dashboard/
│   └── app.py            ← Gradio comparison dashboard
├── evaluation/
│   ├── benchmark.py      ← Runs all 3 pipelines, captures token/cost/latency
│   └── accuracy.py       ← LLM-as-a-Judge + BERTScore evaluation
├── scripts/
│   ├── setup_graphrag.sh ← One-command TigerGraph GraphRAG Docker deploy
│   └── ingest.py         ← Chunk corpus → ChromaDB + TigerGraph GraphRAG
├── configs/
│   └── server_config.json ← TigerGraph GraphRAG config template
├── data/
│   └── test_questions.json ← Evaluation question set with ground truth
├── pyproject.toml
└── .env.example
```

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/your-team/graphrag-hackathon
cd graphrag-hackathon

python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -e ".[dev,notebook]"
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — set LLM_PROVIDER and the matching API key.
# Free options: Groq (llama-3.1-8b-instant) or Gemini (gemini-2.0-flash)
```

### 3. Deploy TigerGraph GraphRAG (for Pipeline 3)

> Requires Docker + Docker Compose Plugin

```bash
chmod +x scripts/setup_graphrag.sh
./scripts/setup_graphrag.sh
# Wait ~2 min for TigerGraph to start
# GraphRAG Chat UI: http://localhost:80
# TigerGraph Studio: http://localhost:14240
```

Copy and edit the LLM config in the deployed service:
```bash
# The setup script creates ./graphrag_service/configs/server_config.json
# Edit the llm_config section to match your LLM provider.
# See configs/server_config.json in this repo for annotated examples.
docker compose -f ./graphrag_service/docker-compose.yml restart graphrag
```

### 4. Ingest your corpus

```bash
# Point at a folder of .txt / .md / .pdf / .jsonl files
# Round 1 requires ≥ 2 million tokens
python scripts/ingest.py --docs /path/to/your/docs/
```

This:
- Chunks every file into ~500-token pieces with 50-token overlap
- Upserts chunks into ChromaDB (for Basic RAG)
- Uploads raw files to the TigerGraph GraphRAG service
- Triggers knowledge graph rebuild (entity extraction + community detection)

### 5. Launch the dashboard

```bash
python -m dashboard.app
# Open http://localhost:7860
```

Enter a query → click **Run All Pipelines** → see side-by-side answers + token/cost/latency metrics.

### 6. Run the full benchmark + accuracy evaluation

```bash
# Step 1: benchmark (tokens, cost, latency)
python -m evaluation.benchmark \
    --questions data/test_questions.json \
    --output    data/benchmark_results.json

# Step 2: accuracy (LLM-as-a-Judge + BERTScore)
python -m evaluation.accuracy \
    --results data/benchmark_results.json \
    --output  data/accuracy_report.json
```

---

## Switching LLM Provider

```bash
# .env
LLM_PROVIDER=groq         # free tier, fast
LLM_PROVIDER=gemini       # free tier, strong
LLM_PROVIDER=openai       # gpt-4o-mini by default
LLM_PROVIDER=anthropic    # claude-haiku-4-5 by default
```

Or override in code:
```python
from pipelines.llm_client import get_client
import pipelines.llm_client as mod
mod.default_client = get_client("groq", model="llama-3.3-70b-versatile")
```

---

## Tuning GraphRAG for Maximum Token Reduction + Accuracy

Follow this order (from the TigerGraph tuning guide):

### 1. Chunking
Start with `chunker: "semantic"`. If your corpus is markdown/PDFs, switch to `"markdown"` with `chunk_size: 2048`, `overlap_size: 256`.

### 2. Entity extraction prompt
Customize via GraphRAG UI: **Settings → Customize Prompts → Entity Relationships**.
Be domain-specific. Example for a legal corpus:
> *"Extract: Case, Court, Judge, Party, Statute, Ruling. Ignore: page numbers, headers, footers."*

### 3. Retrieval parameters

| Question type | top_k | num_hops |
|---|---|---|
| Specific lookup | 3 | 1 |
| Relational | 5 | 2 |
| Broad summarization | 8 | 2 |
| Multi-hop reasoning | 8 | 3 |

Reduce `num_hops` if answers are bloated; increase if answers miss cross-section facts.

### 4. Retriever selection
- `hybrid` (default) — best for most questions
- `community` — best for "summarize the entire topic" questions
- `sibling` — best for "give me more detail on this document" questions

---

## Dataset Recommendations

Pick a domain with **natural entity connections** across documents:

| Domain | Example sources | Why it's good |
|---|---|---|
| Legal | Court cases, statutes | Cases cite other cases (multi-hop) |
| Scientific | arXiv papers | Authors, citations, methods connect across papers |
| News | News articles | People, organizations, events link across stories |
| Medical | Clinical notes, research | Drugs, conditions, treatments are interconnected |
| Support | Ticket archives | Products, issues, resolutions link to each other |

**Round 1 minimum:** 2 million tokens  
**Round 2 (Top 10 only):** 50–100 million tokens, $50 Gemini credits provided

---

## Required Deliverables Checklist

- [ ] **Architecture diagram** — `docs/architecture.png`
- [ ] **Comparison dashboard** — `python -m dashboard.app` (Gradio, port 7860)
- [ ] **Benchmark report** — `data/benchmark_results.json` + `data/accuracy_report.json`
- [ ] **Demo video** — 5–7 min walkthrough showing dashboard live
- [ ] **Public GitHub repo** — built on `github.com/tigergraph/graphrag`
- [ ] **Blog post** — Medium / Hashnode / Dev.to
- [ ] **Social post** — LinkedIn or Twitter, tag @TigerGraph, #GraphRAGInferenceHackathon
- [ ] **Product feedback interview** (Top 5–10 teams only)

---

## Bonus Point Thresholds

| Metric | Threshold | Bonus |
|---|---|---|
| LLM-as-a-Judge pass rate | ≥ 90% | ✅ |
| BERTScore F1 rescaled | ≥ 0.55 | ✅ |
| BERTScore F1 raw | ≥ 0.88 | ✅ |
| Both LLM-Judge ≥ 90% AND BERTScore rescaled ≥ 0.55 | — | 🏆 Maximum bonus |

> Token reduction only counts if GraphRAG maintains or improves accuracy vs Basic RAG.

---

## Useful Links

| Resource | Link |
|---|---|
| TigerGraph GraphRAG Repo | https://github.com/tigergraph/graphrag |
| TigerGraph Savanna (free) | https://tgcloud.io |
| TigerGraph MCP | https://github.com/tigergraph/tigergraph-mcp |
| TigerGraph Docs | https://docs.tigergraph.com |
| Discord | https://discord.gg/4cc7SNqRf |
| WhatsApp Support | https://chat.whatsapp.com/Iwdyhie2gSoIR0k2teMtKb |
| Book a 1:1 | https://calendly.com/devanshu-saxena-tigergraph/20min |