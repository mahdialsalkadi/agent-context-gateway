# syntax=docker/dockerfile:1
# ==============================================================================
# agent-context-gateway -- production image
#
#   docker build -f docker/Dockerfile -t agent-context-gateway .
#   docker run --rm -p 127.0.0.1:8090:8090 \
#     -e UPSTREAM_BASE_URL=https://api.openai.com/v1 \
#     -e UPSTREAM_API_KEY=sk-... \
#     agent-context-gateway
# ==============================================================================
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GATEWAY_HOST=0.0.0.0 \
    GATEWAY_PORT=8090 \
    DATA_DIR=/data \
    LOG_DIR=/logs \
    SHM_CACHE_DIR=/dev/shm/agent_gateway

WORKDIR /app

# Dependencies first so application edits do not invalidate the layer cache.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY pyproject.toml LICENSE README.md ./
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh

# Non-root runtime user. /dev/shm is mounted by the orchestrator; it must be
# writable by this uid, which is why the compose file sets an explicit size.
RUN groupadd --system --gid 10001 gateway \
 && useradd --system --uid 10001 --gid gateway --home-dir /app --shell /usr/sbin/nologin gateway \
 && mkdir -p /data /logs /dev/shm/agent_gateway \
 && chmod +x /usr/local/bin/entrypoint.sh \
 && chown -R gateway:gateway /app /data /logs

USER gateway

EXPOSE 8090

# Uses the standard library so the image needs no extra curl/wget layer.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8090/health', timeout=4).status == 200 else 1)"

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
