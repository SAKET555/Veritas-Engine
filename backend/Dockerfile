# VERITAS API — production image.
#
# State lives in memory and is rebuilt from the hash-chained ledger on boot, so run ONE
# process per data volume (uvicorn --workers 1). Scale out by giving each replica its own
# tenants and volume (shard by X-Tenant-ID at the load balancer), never by pointing two
# processes at the same ledger file.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    AUDIT_LOG_PATH=/data/audit_log.jsonl \
    REPORTS_DIR=/data/reports \
    LLM_PROVIDER=mock

WORKDIR /app
RUN useradd --create-home --uid 10001 veritas \
    && mkdir -p /data \
    && chown veritas:veritas /data

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY *.py ./

USER veritas
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=4s --start-period=20s --retries=3 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/ready', timeout=3).status == 200 else 1)"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers", "--forwarded-allow-ips", "*"]
