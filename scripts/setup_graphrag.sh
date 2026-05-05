#!/usr/bin/env bash
# scripts/setup_graphrag.sh
# --------------------------
# One-command bootstrap for the TigerGraph GraphRAG service.
# Wraps the official setup script from the TigerGraph GraphRAG repo.
#
# Usage:
#   chmod +x scripts/setup_graphrag.sh
#   ./scripts/setup_graphrag.sh
#
# Prerequisites:
#   - Docker + Docker Compose Plugin installed
#   - LLM_PROVIDER and corresponding API key exported in your shell
#   - .env file filled in (copy from .env.example)

set -euo pipefail

source "$(dirname "$0")/../.env" 2>/dev/null || true

GRAPHRAG_DIR="${GRAPHRAG_INSTALL_DIR:-./graphrag_service}"
LLM_PROVIDER="${LLM_PROVIDER:-openai}"

# Pick the right API key to pass to the TigerGraph setup script
case "$LLM_PROVIDER" in
  openai)    export LLM_API_KEY="${OPENAI_API_KEY:-}"  ;;
  gemini|google) export LLM_API_KEY="${GOOGLE_API_KEY:-}" ;;
  anthropic|claude) export LLM_API_KEY="${ANTHROPIC_API_KEY:-}" ;;
  groq)      export LLM_API_KEY="${GROQ_API_KEY:-}" ;;
  *)
    echo "❌ Unknown LLM_PROVIDER: $LLM_PROVIDER"
    exit 1
    ;;
esac

if [[ -z "$LLM_API_KEY" ]]; then
  echo "❌ LLM_API_KEY is empty. Set the correct API key in .env."
  exit 1
fi

echo "▶ Installing TigerGraph GraphRAG service..."
echo "  Provider : $LLM_PROVIDER"
echo "  Directory: $GRAPHRAG_DIR"
echo ""

# Run the official one-step deploy
curl -k https://raw.githubusercontent.com/tigergraph/graphrag/refs/heads/main/docs/tutorials/setup_graphrag.sh \
  | bash -s -- "$GRAPHRAG_DIR" "$LLM_PROVIDER"

echo ""
echo "✅ GraphRAG service deployed at: $GRAPHRAG_DIR"
echo "   TigerGraph UI : http://localhost:14240"
echo "   GraphRAG Chat : http://localhost:80"
echo ""
echo "Next steps:"
echo "  1. Wait ~2 minutes for TigerGraph to fully start"
echo "  2. Run:  ./scripts/ingest.sh <path/to/your/docs/>"
echo "  3. Run:  python -m dashboard.app"