# Grading dashboard: the React frontend built to static files, served by the FastAPI backend.
# One container, one origin: the page and its /api calls come from the same place.
#
# Build args (baked into the frontend at build time):
#   VITE_BASE          URL path the dashboard is served under. Default "/". Must end in "/".
#   VITE_API_BASE_URL  Only if the API is on a different origin. Default: same origin.

# ---- Stage 1: build the frontend ----
FROM node:20.19-alpine AS frontend
WORKDIR /frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
ARG VITE_BASE=/
ARG VITE_API_BASE_URL=
ENV VITE_BASE=$VITE_BASE \
    VITE_API_BASE_URL=$VITE_API_BASE_URL
RUN npm run build

# ---- Stage 2: backend runtime ----
FROM python:3.12-slim
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STATIC_DIR=/app/static \
    DATA_PATH=/data/extracted_chats.json \
    OUTPUT_PATH=/data/extracted_chats.json

COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./
COPY --from=frontend /frontend/dist ./static

# Run unprivileged. /data is created here so a fresh named volume inherits its ownership.
RUN useradd --system --uid 10001 grading \
    && mkdir -p /data \
    && chown grading /data
USER grading

EXPOSE 9500
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9500"]
