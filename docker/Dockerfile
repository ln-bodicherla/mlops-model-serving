FROM python:3.11-slim as builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /install /usr/local

RUN groupadd -r appuser && useradd -r -g appuser appuser

COPY src/ ./src/
COPY configs/ ./configs/

RUN mkdir -p models logs && chown -R appuser:appuser /app

USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CONFIG_PATH=/app/configs/config.yaml \
    MODEL_PATH=/app/models/champion.joblib

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/model/health'); exit(0 if r.status_code == 200 else 1)"

CMD ["uvicorn", "src.model_server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "--log-level", "info"]
