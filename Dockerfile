# One image for every CloudRAG service; the service is chosen by the command argument:
#   docker run cloudrag:0.1.0 gateway | ingestion | embedding | indexing | retrieval | reranker | generation | all
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CLOUDRAG_DATA_DIR=/data

WORKDIR /app
COPY pyproject.toml README.md ./
COPY cloudrag ./cloudrag
RUN pip install ".[redis,anthropic]" \
 && useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin cloudrag \
 && mkdir -p /data && chown 10001:10001 /data

USER 10001
EXPOSE 8000 8001 8002 8003 8004 8005 8006
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/healthz', timeout=2)"
ENTRYPOINT ["cloudrag", "serve"]
CMD ["all"]
