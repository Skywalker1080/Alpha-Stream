# syntax=docker/dockerfile:1

ARG UV_VERSION=0.12.5

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# ---------- Builder stage ----------
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

# Layer 1: dependencies. Copied first so app.py edits never re-run the
# (slow) streamlit/pandas install — only requirements.txt changes bust it.
COPY requirements.txt /build/requirements.txt
RUN uv venv /opt/venv \
    && uv pip install --python /opt/venv/bin/python --no-cache \
       -r /build/requirements.txt

# ---------- Runtime stage ----------
FROM python:3.13-slim-bookworm

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

WORKDIR /frontend

RUN groupadd --system app \
    && useradd --system --gid app --create-home app \
    && mkdir -p /frontend \
    && chown -R app:app /frontend

COPY --from=builder /opt/venv /opt/venv

# Application code last — cached layers above are reused across builds.
COPY app.py /frontend/app.py

USER app

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD python -c "import socket; socket.create_connection(('127.0.0.1', 8501), 3)" || exit 1

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]