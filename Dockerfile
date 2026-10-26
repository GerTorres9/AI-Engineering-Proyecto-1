FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY data ./data
COPY scripts ./scripts
COPY pyproject.toml README.md ./

RUN mkdir -p /app/storage /app/results

EXPOSE 8000

CMD ["uvicorn", "intelligence_system.api:app", "--host", "0.0.0.0", "--port", "8000"]
