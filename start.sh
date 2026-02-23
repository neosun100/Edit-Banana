#!/bin/bash
set -e

echo "🍌 Starting Edit Banana..."

# GPU auto-select
if command -v nvidia-smi &>/dev/null; then
    GPU_ID=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
        sort -t',' -k2 -n | head -1 | cut -d',' -f1)
    export NVIDIA_VISIBLE_DEVICES=$GPU_ID
    echo "📊 Selected GPU $GPU_ID (least memory used)"
    nvidia-smi -i $GPU_ID --query-gpu=name,memory.total,memory.used --format=csv,noheader
fi

PORT=${PORT:-8450}
MCP_PORT=${MCP_PORT:-8452}

# Start SAM3 service in background (port 8451)
echo "🔧 Starting SAM3 service on :8451..."
python3 sam3_service/server.py --port 8451 --config config/config.yaml &
SAM3_PID=$!

# Start MCP server in background
echo "🔧 Starting MCP server on :$MCP_PORT..."
python3 mcp_server.py &
MCP_PID=$!

# Start main app
echo "🌐 Starting UI + API on :$PORT..."
python3 app.py

# Cleanup
kill $SAM3_PID $MCP_PID 2>/dev/null || true
