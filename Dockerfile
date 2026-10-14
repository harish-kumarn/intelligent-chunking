# Vision RAG service — one container, one process, one worker.
#
#   docker build -t vision-rag .
#   docker run --env-file .env -p 8000:8000 vision-rag

FROM python:3.12-slim

WORKDIR /srv

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
# Pipeline choices that are not secrets (OCR engine switch). Bind-mount over it
# to change them without a rebuild, or override any key with an env var.
COPY config.yaml ./config.yaml

ENV ARTIFACT_MODE=debug \
    SERVER_HOST=0.0.0.0 \
    SERVER_PORT=8000 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/v1/health', timeout=3).status == 200 else 1)"

# Exactly one worker: the in-process admission gate (app/concurrency.py) assumes
# a single process handles at most one document at a time.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
