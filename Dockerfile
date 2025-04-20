# ================ BUILD STAGE ================
FROM python:3.12-slim AS builder

# Set work directory for the build stage
WORKDIR /app

# Set environment variables to optimize Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir wheel \
    && pip install --no-cache-dir -r requirements.txt

# ================ RUNTIME STAGE ================
FROM python:3.12-slim

# Set work directory for the runtime stage
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONOPTIMIZE=2 \
    RUST_BACKTRACE=0 \
    APP_ENV=production

# Create a non-root user with proper home directory
RUN addgroup --system --gid 1001 app && \
    adduser --system --uid 1001 --gid 1001 --home /home/app app && \
    mkdir -p /app/logs /app/data /home/app && \
    chown -R app:app /app /home/app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed dependencies from builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the project code
COPY --chown=app:app . .

# Create directories and set permissions
RUN mkdir -p /app/logs /app/data \
    && chmod -R 755 /app/logs /app/data \
    && chown -R app:app /app/logs /app/data

# Switch to non-root user
USER app

# Create an entrypoint script
RUN echo '#!/bin/sh\n\
mkdir -p /app/logs\n\
python -m app.main' > /app/entrypoint.sh \
    && chmod +x /app/entrypoint.sh

# Set up healthcheck
HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
    CMD curl --fail http://localhost:9005/ || exit 1

# Expose the port the app runs on
EXPOSE 9005

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]