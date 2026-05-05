#!/usr/bin/env python3
"""
scripts/ingest.py
------------------
Ingest your corpus into both:
  1. ChromaDB  — for Basic RAG (Pipeline 2)
  2. TigerGraph GraphRAG service — for GraphRAG (Pipeline 3)

Supports: .txt, .md, .pdf (via pdfminer), .jsonl (one doc per line)

Usage:
    python scripts/ingest.py --docs path/to/docs/ [--chroma-dir ./data/chroma_db]

The script:
  - Walks the docs directory recursively
  - Splits each file into ~500-token chunks with 50-token overlap
  - Upserts chunks into ChromaDB (for Basic RAG)
  - Uploads raw files to TigerGraph GraphRAG (for GraphRAG)
  - Triggers a knowledge graph rebuild

Dataset requirement (Round 1): at least 2 million tokens of text.
"""

from __future__ import annotations

import os
import sys
import re
import argparse
import hashlib
from pathlib import Path

from tqdm import tqdm
import tiktoken

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.basic_rag import ingest_documents
from pipelines.graphrag  import ingest_documents_to_graphrag, init_knowledge_graph, rebuild_knowledge_graph

CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", "500"))    # tokens
OVERLAP_SIZE  = int(os.getenv("OVERLAP_SIZE", "50"))   # tokens
GRAPHRAG_GRAPH = os.getenv("GRAPHRAG_GRAPH", "MyGraph")


# ── Tokenizer ─────────────────────────────────────────────────────────────────

_enc = tiktoken.get_encoding("cl100k_base")

def tokenize(text: str) -> list[int]:
    return _enc.encode(text)

def detokenize(tokens: list[int]) -> str:
    return _enc.decode(tokens)


# ── File readers ──────────────────────────────────────────────────────────────

def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def read_pdf(path: Path) -> str:
    try:
        from pdfminer.high_level import extract_text
        return extract_text(str(path))
    except ImportError:
        print("  ⚠️  pdfminer not installed. Install with: pip install pdfminer.six")
        return ""


def read_jsonl(path: Path) -> str:
    """Concatenate all 'text' or 'content' fields from a JSONL file."""
    import json
    lines = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        text = obj.get("text") or obj.get("content") or obj.get("body") or ""
        if text:
            lines.append(text)
    return "\n\n".join(lines)


def read_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return read_pdf(path)
    elif ext in (".jsonl", ".json"):
        return read_jsonl(path)
    else:  # .txt, .md, etc.
        return read_txt(path)


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP_SIZE,
) -> list[str]:
    """
    Split text into overlapping token-level chunks.
    Returns a list of decoded text chunks.
    """
    tokens = tokenize(text)
    chunks = []
    start  = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(detokenize(tokens[start:end]))
        if end == len(tokens):
            break
        start += chunk_size - overlap
    return chunks


# ── Ingestion ─────────────────────────────────────────────────────────────────

def ingest_directory(
    docs_dir: Path,
    chroma_dir: str = "./data/chroma_db",
    upload_to_graphrag: bool = True,
    graphrag_graph: str = GRAPHRAG_GRAPH,
) -> dict:
    """
    Walk docs_dir, chunk every supported file, and ingest into both systems.
    """
    supported = {".txt", ".md", ".pdf", ".jsonl", ".json", ".rst"}
    file_paths = [
        p for p in docs_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in supported
    ]

    if not file_paths:
        print(f"⚠️  No supported files found in {docs_dir}")
        return {}

    print(f"Found {len(file_paths)} files.")

    # ── ChromaDB ingestion (Basic RAG) ────────────────────────────────────────
    chroma_docs = []
    total_tokens = 0

    for fpath in tqdm(file_paths, desc="Chunking for ChromaDB"):
        text = read_file(fpath)
        if not text.strip():
            continue
        chunks = chunk_text(text)
        total_tokens += sum(len(tokenize(c)) for c in chunks)

        for i, chunk in enumerate(chunks):
            doc_id = hashlib.md5(f"{fpath.name}_{i}".encode()).hexdigest()
            chroma_docs.append({
                "id":   doc_id,
                "text": chunk,
                "metadata": {
                    "source":      fpath.name,
                    "path":        str(fpath),
                    "chunk_index": i,
                },
            })

    print(f"\nIngesting {len(chroma_docs)} chunks into ChromaDB...")
    ingested = ingest_documents(chroma_docs, persist_dir=chroma_dir)
    print(f"✅ ChromaDB: {ingested} chunks ingested ({total_tokens:,} total tokens)")

    # ── TigerGraph GraphRAG ingestion ─────────────────────────────────────────
    graphrag_result = {}
    if upload_to_graphrag:
        print(f"\nInitializing TigerGraph graph '{graphrag_graph}'...")
        try:
            init_result = init_knowledge_graph(graph=graphrag_graph)
            print(f"  Init: {init_result}")

            print(f"Uploading {len(file_paths)} files to GraphRAG service...")
            graphrag_result = ingest_documents_to_graphrag(
                file_paths=[str(p) for p in file_paths],
                graph=graphrag_graph,
            )
            print(f"  Upload: {graphrag_result}")

            print("Triggering knowledge graph rebuild (this may take a few minutes)...")
            rebuild_result = rebuild_knowledge_graph(graph=graphrag_graph)
            print(f"  Rebuild: {rebuild_result}")
            print("✅ TigerGraph GraphRAG: ingestion and rebuild complete")

        except Exception as e:
            print(f"⚠️  TigerGraph GraphRAG upload failed: {e}")
            print("   Make sure the GraphRAG service is running (./scripts/setup_graphrag.sh)")

    # ── Token count summary ───────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"Dataset summary:")
    print(f"  Files processed : {len(file_paths)}")
    print(f"  Total chunks    : {len(chroma_docs)}")
    print(f"  Total tokens    : {total_tokens:,}")
    if total_tokens < 2_000_000:
        shortage = 2_000_000 - total_tokens
        print(f"  ⚠️  Round 1 requires ≥ 2M tokens. You're {shortage:,} tokens short.")
    else:
        print(f"  ✅ Meets Round 1 minimum (≥ 2M tokens)")
    print(f"{'='*50}\n")

    return {
        "files":         len(file_paths),
        "chunks":        len(chroma_docs),
        "total_tokens":  total_tokens,
        "graphrag":      graphrag_result,
    }


def main():
    parser = argparse.ArgumentParser(description="Ingest corpus into ChromaDB + TigerGraph GraphRAG")
    parser.add_argument("--docs",         required=True, help="Path to docs directory")
    parser.add_argument("--chroma-dir",   default="./data/chroma_db")
    parser.add_argument("--graph",        default=GRAPHRAG_GRAPH)
    parser.add_argument("--no-graphrag",  action="store_true", help="Skip TigerGraph upload")
    args = parser.parse_args()

    docs_dir = Path(args.docs)
    if not docs_dir.exists():
        print(f"❌ Directory not found: {docs_dir}")
        sys.exit(1)

    ingest_directory(
        docs_dir=docs_dir,
        chroma_dir=args.chroma_dir,
        upload_to_graphrag=not args.no_graphrag,
        graphrag_graph=args.graph,
    )


if __name__ == "__main__":
    main()