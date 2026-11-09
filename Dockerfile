# Dispatcher console: builds frontend/, serves it with backend/ on port 8000. Needs ANTHROPIC_API_KEY at run time.
FROM node:22-alpine AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.13-slim
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/*.py ./
COPY --from=web /web/dist ./static
ENV STATIC=/app/static BOARD=/app/static/board.json
EXPOSE 8000
# Behind a hosting proxy set FORWARDED_ALLOW_IPS to its address, so the per-IP limit sees real visitors.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
