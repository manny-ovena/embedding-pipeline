#!/usr/bin/env bash
# Ensures a vLLM server is running for the Llama-3.2-1B model, then runs the
# llm-runtime crate.
set -euo pipefail

MODEL_PATH="/srv/ai-models/Qwen/Qwen3-Embedding-4B"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-Embedding-4B}"
HOST="0.0.0.0"
PORT="8000"
VENV_DIR="/home/mannyo/venvs/vllm"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

is_vllm_running() {
    curl --silent --fail --max-time 2 "http://localhost:${PORT}/health" >/dev/null 2>&1
}

if is_vllm_running; then
    echo "vLLM server already running on port ${PORT}."
else
    echo "Starting vLLM server for ${MODEL_PATH}..."
    export VLLM_USE_V2_MODEL_RUNNER=0
    "$VENV_DIR/bin/vllm" serve "$MODEL_PATH" \
        --host "$HOST" \
        --port "$PORT" \
        --served-model-name "$SERVED_MODEL_NAME" \
        --generation-config vllm \
        --enable-prompt-embeds &
    VLLM_PID=$!

    echo "Waiting for vLLM server to become ready (pid ${VLLM_PID})..."
    until is_vllm_running; do
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "vLLM server process exited unexpectedly." >&2
            exit 1
        fi
        sleep 2
    done
    echo "vLLM server is ready."
fi
