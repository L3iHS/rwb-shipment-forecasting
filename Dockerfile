# Dispatcher console: builds frontend/, serves it with backend/ and the service model on port 8000.
# Local all-in-one image. Production uses deploy/compose.yml and the two deploy/*.Dockerfile images.
# The assistant needs ROUTERAI_API_KEY at run time; the file forecast works without it.
FROM node:22-alpine AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.13-slim
WORKDIR /app
# LightGBM needs the OpenMP runtime, which the slim image does not have.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/*.py ./
COPY --from=web /web/dist ./static
# The service model and the history it forecasts from, for the "Свой файл" tab (/api/forecast).
COPY src ./src
COPY models/prod/service ./models/prod/service
COPY data/raw ./data/raw
ENV STATIC=/app/static BOARD=/app/static/board.json RWB_ROOT=/app
EXPOSE 8000
# Behind a hosting proxy set FORWARDED_ALLOW_IPS to its address, so the per-IP limit sees real visitors.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
