# ============================================================
# JCAB / Adaptive Assessment - Production Dockerfile
# ============================================================
# Multi-stage/optimized lightweight production container
# Includes pre-cached models, non-root user, and health checks

FROM python:3.11-slim

# Set environment flags
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/home/appuser/.cache/huggingface \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true

# Install system runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -g 1000 appuser && \
    useradd -u 1000 -g appuser -m -s /bin/bash appuser

WORKDIR /app

# Install PyTorch CPU wheel first to minimize image size (avoids 2.5GB CUDA bloat)
RUN pip install --no-cache-dir torch==2.1.2+cpu torchvision==0.16.2+cpu --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-cache models into /home/appuser/.cache/huggingface
COPY deploy/scripts/precache_models.py /tmp/precache_models.py
RUN chown -R appuser:appuser /home/appuser && \
    su appuser -c "python /tmp/precache_models.py" && \
    rm /tmp/precache_models.py

# Copy application source code
COPY . /app

# Ensure directories for logs and persistent data exist with proper permissions
RUN mkdir -p /app/logs /app/research/validation && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose Streamlit port
EXPOSE 8501

# Healthcheck using Streamlit internal health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Launch application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
