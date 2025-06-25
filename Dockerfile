# OpenInferencev2 Production Dockerfile
# Multi-stage build for optimal performance and security

# Build stage
FROM nvidia/cuda:11.8-devel-ubuntu22.04 AS builder

# Set environment variables for build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    build-essential \
    cmake \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

# Copy source code and install
COPY . /build/
WORKDIR /build
RUN pip install -e .

# Production stage
FROM nvidia/cuda:11.8-runtime-ubuntu22.04 AS production

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/opt/venv/bin:$PATH"
ENV CUDA_VISIBLE_DEVICES=all
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    curl \
    htop \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r openinference && useradd -r -g openinference openinference

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY --from=builder /build /app
WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/logs /app/models /app/cache \
    && chown -R openinference:openinference /app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD python -c "import openinferencev2; print('OK')" || exit 1

# Switch to non-root user
USER openinference

# Expose ports
EXPOSE 8000 9090

# Default command
CMD ["python", "-m", "src.cli.main", "--help"]

# Development stage
FROM builder AS development

# Install development dependencies
COPY requirements-dev.txt /tmp/
RUN pip install -r /tmp/requirements-dev.txt

# Install additional development tools
RUN apt-get update && apt-get install -y \
    vim \
    tmux \
    fish \
    && rm -rf /var/lib/apt/lists/*

# Copy development configuration
COPY .pre-commit-config.yaml /build/
COPY Makefile /build/

# Set development environment
ENV ENVIRONMENT=development
ENV DEBUG=1

WORKDIR /build

# Development command
CMD ["bash"] 