#!/usr/bin/env bash
# Stops any running vLLM server processes safely.
set -euo pipefail

PIDS=$(pgrep -f "vllm serve" || true)

if [ -z "$PIDS" ]; then
    echo "No running vLLM server processes found."
    exit 0
fi

echo "Stopping vLLM process(es): $PIDS"
for PID in $PIDS; do
    echo "Terminating PID $PID..."
    kill "$PID" || true
done

# Wait briefly and verify if processes exited
sleep 2

REMAINING_PIDS=$(pgrep -f "vllm serve" || true)
if [ -n "$REMAINING_PIDS" ]; then
    echo "vLLM process(es) still running ($REMAINING_PIDS). Sending SIGKILL..."
    for PID in $REMAINING_PIDS; do
        kill -9 "$PID" 2>/dev/null || true
    done
fi

echo "vLLM server stopped."
