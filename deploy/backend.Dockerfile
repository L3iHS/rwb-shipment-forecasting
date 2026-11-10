FROM python:3.13-slim AS dependencies
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RWB_ROOT=/app \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

FROM dependencies AS prepared
COPY src/ /app/src/
COPY backend/runtime.py /app/runtime.py
COPY data/raw/ /app/data/raw/
RUN python runtime.py

FROM dependencies AS runtime
RUN groupadd --gid 10001 rwb && useradd --uid 10001 --gid 10001 --no-create-home rwb \
    && mkdir -p /var/cache/rwb && chown 10001:10001 /var/cache/rwb
COPY backend/*.py /app/
COPY src/ /app/src/
COPY models/prod/service/ /app/models/prod/service/
COPY frontend/public/board.json /app/static/board.json
COPY --from=prepared /app/prepared/ /app/prepared/
ENV BOARD=/app/static/board.json STATIC=/app/static RWB_PREPARED=/app/prepared RWB_SCRATCH=/var/cache/rwb
USER 10001:10001
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]

FROM runtime AS test
USER root
RUN pip install --no-cache-dir pytest==9.2.0 httpx==0.28.1
COPY tests/test_upload.py tests/test_runtime.py /app/tests/
COPY frontend/public/demo/ /app/tests/demo/
USER 10001:10001
CMD ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "/app/tests/test_upload.py", "/app/tests/test_runtime.py"]
