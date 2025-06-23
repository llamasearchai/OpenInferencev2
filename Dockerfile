FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    pkg-config \
    libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
COPY setup.py .
COPY tox.ini .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir tox

# Copy project files
COPY . .

# Install the package in development mode
RUN pip install -e .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash openinferencev2
RUN chown -R openinferencev2:openinferencev2 /app
USER openinferencev2

# Set default command
CMD ["python", "-m", "pytest", "tests/", "-v"]

# Expose port for CLI interface
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import openinferencev2; print('OpenInferencev2 is healthy')" || exit 1 