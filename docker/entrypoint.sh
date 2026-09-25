#!/bin/sh
# ==============================================================================
# Container entrypoint.
#
# The gateway and its maintenance daemon must see the SAME cache directory. A
# tmpfs such as /dev/shm is per-container, so running the sentinel as a second
# container would leave it unable to prune the gateway's spill files. Starting
# both processes here keeps them on one tmpfs.
# ==============================================================================
set -eu

: "${GATEWAY_HOST:=0.0.0.0}"
: "${GATEWAY_PORT:=8090}"
: "${SENTINEL_INTERVAL:=86400}"

mkdir -p "${DATA_DIR:-/data}" "${LOG_DIR:-/logs}" "${SHM_CACHE_DIR:-/dev/shm/agent_gateway}" 2>/dev/null || true

# The sentinel is a janitor; if it dies the gateway must keep serving.
python -m src.sentinel --loop --interval "$SENTINEL_INTERVAL" \
    >>"${LOG_DIR:-/logs}/sentinel.log" 2>&1 &

exec python -m uvicorn src.gateway:app --host "$GATEWAY_HOST" --port "$GATEWAY_PORT"
