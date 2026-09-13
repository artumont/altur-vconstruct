FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV UV_PROJECT_ENVIRONMENT="/opt/venv"
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/opt/hf-cache

COPY --from=train . ./train
COPY --from=train ./checkpoints/ ./model/checkpoints/
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Download WavLM-large and export to ONNX (single-use, no runtime transformers dep)
RUN uv pip install transformers onnx && python -c "\
import torch; \
from transformers import WavLMModel; \
print('Downloading WavLM-large...'); \
m = WavLMModel.from_pretrained('microsoft/wavlm-large'); \
m.eval(); \
print('Exporting to ONNX...'); \
torch.onnx.export( \
    m, torch.randn(1, 48000), \
    'model/checkpoints/wavlm-large.onnx', \
    input_names=['input_values'], \
    output_names=['last_hidden_state'], \
    dynamic_axes={'input_values': {0: 'batch'}, 'last_hidden_state': {0: 'batch'}}, \
    opset_version=14, \
    dynamo=False \
); \
print('Done') \
"

FROM python:3.12-slim

RUN addgroup --system app && adduser --system --ingroup app app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app/train /app/train
COPY --from=builder /app/model /app/model

WORKDIR /app
COPY ./src .

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app/src:/app/train"

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
