FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV UV_PROJECT_ENVIRONMENT="/opt/venv"
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY --from=train . ./train
COPY --from=train ./checkpoints/ ./model/checkpoints/
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Pre-download WavLM-large so the runtime image never hits the HF Hub
ENV HF_HOME=/opt/hf-cache
RUN python -c "from transformers import WavLMModel; WavLMModel.from_pretrained('microsoft/wavlm-large')"

FROM python:3.12-slim

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app/train /app/train
COPY --from=builder /app/model /app/model
COPY --from=builder /opt/hf-cache /opt/hf-cache

WORKDIR /app
COPY ./src .

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app/src:/app/train"
ENV HF_HOME=/opt/hf-cache

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
