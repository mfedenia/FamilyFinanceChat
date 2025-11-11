FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy backend files (make sure all .py files are copied)
COPY backend/requirements.txt backend/requirements.txt
COPY backend/*.py backend/
COPY backend/Webscraping backend/Webscraping

# Install Python dependencies
RUN pip install --no-cache-dir -r backend/requirements.txt && \
    pip install playwright beautifulsoup4 pymupdf tqdm python-dotenv && \
    playwright install chromium && \
    playwright install-deps

# Copy frontend build (we'll build it separately)
COPY frontend/upload/upload-app/build frontend/build

# Expose port
EXPOSE 8000

# Set PYTHONPATH to include backend directory
ENV PYTHONPATH=/app/backend:$PYTHONPATH

# Start the backend
CMD ["python", "-m", "uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]