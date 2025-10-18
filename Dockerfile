# Multi-stage Dockerfile for ARA AI
# Supports Linux, Windows, and macOS builds

FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    TF_ENABLE_ONEDNN_OPTS=0

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copy application code
COPY . .

# Install the package
RUN pip install -e .

# Create models directory
RUN mkdir -p models

# Expose port for potential web interface
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from meridianalgo.ultimate_ml import UltimateStockML; print('OK')" || exit 1

# Default command
CMD ["python", "ara.py", "--help"]


# Development stage
FROM base as development

# Install development dependencies
RUN pip install pytest pytest-cov pytest-xdist black flake8 pylint

# Run tests by default in dev
CMD ["pytest", "tests/", "-v"]


# Production stage
FROM base as production

# Remove unnecessary files
RUN find . -type d -name __pycache__ -exec rm -rf {} + && \
    find . -type f -name "*.pyc" -delete && \
    find . -type f -name "*.pyo" -delete

# Run as non-root user
RUN useradd -m -u 1000 arauser && \
    chown -R arauser:arauser /app

USER arauser

# Production command
CMD ["python", "ara_fast.py", "AAPL"]
