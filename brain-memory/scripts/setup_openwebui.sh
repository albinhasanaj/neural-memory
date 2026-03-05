#!/usr/bin/env bash
# ── setup_openwebui.sh ─────────────────────────────────────────────
# Download and run Open WebUI with Docker, configured to use the
# brain-memory proxy server as its backend.
#
# Prerequisites: Docker must be installed and running.
# Usage: bash scripts/setup_openwebui.sh

set -euo pipefail

PROXY_HOST="${BRAIN_PROXY_HOST:-localhost}"
PROXY_PORT="${BRAIN_PROXY_PORT:-8800}"

echo "=== Brain Memory — Open WebUI Setup ==="
echo ""
echo "This script will:"
echo "  1. Pull the Open WebUI Docker image"
echo "  2. Run it with the brain-memory proxy as the OpenAI backend"
echo ""
echo "Proxy endpoint: http://${PROXY_HOST}:${PROXY_PORT}/v1"
echo ""

# Pull latest Open WebUI
echo "[1/2] Pulling Open WebUI Docker image..."
docker pull ghcr.io/open-webui/open-webui:main

# Run Open WebUI pointing at the brain-memory proxy
echo "[2/2] Starting Open WebUI container..."
docker run -d \
  --name brain-memory-webui \
  --network host \
  -e OPENAI_API_BASE_URL="http://${PROXY_HOST}:${PROXY_PORT}/v1" \
  -e OPENAI_API_KEY="brain-memory-proxy" \
  -e WEBUI_AUTH=false \
  -v open-webui-data:/app/backend/data \
  ghcr.io/open-webui/open-webui:main

echo ""
echo "=== Setup complete! ==="
echo ""
echo "  Open WebUI:    http://localhost:8080"
echo "  Brain Proxy:   http://${PROXY_HOST}:${PROXY_PORT}"
echo ""
echo "Make sure the brain-memory proxy is running:"
echo "  brain-proxy"
echo "  # or: uvicorn pipeline.proxy_server:app --host 0.0.0.0 --port ${PROXY_PORT}"
echo ""
echo "To stop: docker stop brain-memory-webui && docker rm brain-memory-webui"
