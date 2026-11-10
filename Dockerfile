FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir .

COPY . .
RUN chmod +x scripts/entrypoint.sh

EXPOSE 8000

CMD ["./scripts/entrypoint.sh"]
