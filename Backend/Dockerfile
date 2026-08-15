# syntax=docker/dockerfile:1

ARG UV_VERSION=0.12.5

# uv image provides the pinned uv binary; FROM (unlike COPY --from) supports
# ARG expansion, so we stage it here first.
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# ---------- Builder stage ----------
# Python dependencies are resolved into a dedicated venv here so the runtime
# stage stays slim (no build toolchain) and the huge torch/feast layer is
# cached until Backend/requirements.txt or timesfm/ changes.
FROM python:3.13-slim-bookworm AS builder

ARG UV_VERSION
ENV UV_VERSION=${UV_VERSION} \
    PATH="/uv/bin:$PATH" \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_SYSTEM_PYTHON=0 \
    PYTHONUNBUFFERED=1

WORKDIR /build

COPY --from=uv /uv /uv/bin/uv

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

# Layer 1: Python dependencies. Copied first so a code change never re-runs
# this expensive install; only editing requirements.txt invalidates it.
COPY Backend/requirements.txt /build/Backend/requirements.txt
RUN uv venv /opt/venv \
    && uv pip install --python /opt/venv/bin/python --no-cache \
       -r /build/Backend/requirements.txt

# Layer 2: the LOCAL timesfm v2.5 package (overrides PyPI v1; provides
# TimesFM_2p5_200M_torch). Cached until timesfm/ changes.
COPY timesfm/ /build/timesfm/
RUN uv pip install --python /opt/venv/bin/python --no-cache /build/timesfm

# ---------- Runtime stage ----------
FROM python:3.13-slim-bookworm

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app" \
    ORT_LOGGING_LEVEL=ERROR \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Non-root user. NOTE: /app is bind-mounted from the host under compose, so
# the container's writes to outputs/, mlruns/ etc. inherit host ownership.
RUN groupadd --system app \
    && useradd --system --gid app --create-home app \
    && mkdir -p /app \
    && chown -R app:app /app

# Pre-built venv from the builder stage.
COPY --from=builder /opt/venv /opt/venv

# Application code last — code edits reuse every cached layer above.
# The TimesFM weights (model/) are intentionally NOT baked in; compose mounts
# the repo root into /app so they resolve at runtime.
COPY src/ /app/src/
COPY logger/ /app/logger/
COPY Backend/ /app/Backend/

USER app

EXPOSE 8000

# python:3.13-slim ships no curl/wget, so probe the port with the stdlib.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import socket; socket.create_connection(('127.0.0.1', 8000), 3)" || exit 1

CMD ["bash", "Backend/start.sh"]