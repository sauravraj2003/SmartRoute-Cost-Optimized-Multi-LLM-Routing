FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY gateway gateway
COPY router router
COPY features features
COPY benchmark benchmark
COPY data data
COPY eval eval
RUN pip install --no-cache-dir .
COPY configs configs
EXPOSE 8000
CMD ["uvicorn", "gateway.api:app", "--host", "0.0.0.0", "--port", "8000"]
