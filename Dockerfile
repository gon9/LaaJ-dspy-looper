# syntax=docker/dockerfile:1

# Stage 1: Build & Dependencies
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# 依存関係ファイルのコピー
COPY pyproject.toml uv.lock ./

# 依存関係のインストール（仮想環境 /app/.venv に作成）
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --all-extras

# ソースコードのコピーとプロジェクトのインストール
COPY src/ ./src/
COPY README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --all-extras

# Stage 2: Minimal Runtime
FROM python:3.12-slim-bookworm AS runtime

WORKDIR /app

# 非rootユーザーの作成
RUN useradd -m -u 1000 appuser

# builder から仮想環境をコピー
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --chown=appuser:appuser . /app/

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

USER appuser

ENTRYPOINT ["laaj"]
CMD ["--help"]
