# Reference image for the Postgres deployment (see docker-compose.yml).
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.27 /uv /bin/uv

# Build the venv outside /app so the compose bind mount (./:/app) cannot shadow it.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --extra postgres --extra llm --extra gui

# CLI, not a daemon: default to help. No ENTRYPOINT, so compose's
# `command: [sleep, infinity]` and `run --rm dexta dexta init` both work.
CMD ["dexta", "--help"]
